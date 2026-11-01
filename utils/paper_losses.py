import math
import warnings
import torch
import torch.nn.functional as F
from evaluation.rmsd import kabsch_torch
from utils.loss_config import resolve_loss_config

def _require_finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Technical failure: nonfinite ' + name)

def _count_by_complex(residue_indices, batch_id, batch_size):
    return torch.bincount(batch_id[residue_indices], minlength=batch_size)

def _connected_zero(value):
    return value.reshape(-1)[:0].sum()

def _finite_sum(values):
    result = values.sum()
    if not bool(torch.isfinite(result)):
        _require_finite(values, 'loss reduction inputs')
        result = values.sum(dtype=torch.float64)
        _require_finite(result, 'loss reduction')
    return result

def _sum_count(loss_values, count, connected_to):
    return (_finite_sum(loss_values) + _connected_zero(connected_to)) / count.clamp_min(1)

def _loss_precision_risk(pred_X, true_X, observed_occupied, movable, protocol, shadow_X, paratope_mask, active_terms):
    if pred_X.dtype == torch.float64:
        return False

    def maximum(values):
        return float(values.detach().abs().max()) if values.numel() else 0.0
    selected = observed_occupied & movable[:, None]
    pred_max = maximum(pred_X[selected])
    ref_max = maximum(true_X[observed_occupied])
    shadow_max = 0.0
    if 'shadow' in active_terms and shadow_X is not None and (paratope_mask is not None):
        mask = observed_occupied[paratope_mask.bool()]
        if shadow_X.shape[:2] == mask.shape:
            shadow_max = maximum(shadow_X[mask])
    safe_square = math.sqrt(torch.finfo(torch.float32).max) / 16.0
    if pred_max + ref_max + shadow_max > safe_square:
        return True
    if protocol != 'P1':
        points = int(selected[:, :4].sum())
        if 12.0 * points * ref_max * pred_max > safe_square:
            return True
    return False

def _bond_lengths(atoms):
    result = torch.linalg.vector_norm(atoms[:, 0] - atoms[:, 1], dim=-1)
    if not bool(torch.isfinite(result).all()):
        _require_finite(atoms, 'bond endpoints')
        result = torch.linalg.vector_norm(atoms[:, 0].double() - atoms[:, 1].double(), dim=-1)
        _require_finite(result, 'bond lengths')
    return result

def reference_coordinate_eligibility(true_X, observed_occupied, movable_mask, batch_id, batch_size=None):
    if batch_size is None:
        batch_size = int(batch_id.max().item()) + 1 if batch_id.numel() else 0
    result = torch.zeros(batch_size, dtype=torch.bool, device=true_X.device)
    with torch.no_grad():
        for b in range(batch_size):
            residues = (batch_id == b) & movable_mask
            points = true_X[residues, :4][observed_occupied[residues, :4]]
            if len(points) < 3:
                continue
            _require_finite(points, 'observed reference backbone')
            points = points.double()
            singular = torch.linalg.svdvals(points - points.mean(dim=0))
            result[b] = (singular[0] > 0) & (singular[1] > 1e-06 * singular[0])
    return result

def _bond_term(pred_X, true_X, given_X, movable, occupied, observed, batch_id, batch_size, bonds):
    n, channels = pred_X.shape[:2]
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError('Physical bonds must have shape [E, 2]')
    if bonds.numel() and (bool((bonds < 0).any()) or bool((bonds >= n * channels).any())):
        raise ValueError('Physical bond atom index out of range')
    left, right = (bonds[:, 0], bonds[:, 1])
    residue_left, residue_right = (left // channels, right // channels)
    if bool((batch_id[residue_left] != batch_id[residue_right]).any()):
        raise ValueError('Physical bonds must not cross complexes')
    flat_occupied = occupied.reshape(-1)
    flat_observed = observed.reshape(-1)
    flat_movable = movable[:, None].expand(n, channels).reshape(-1)
    candidate = (flat_movable[left] | flat_movable[right]) & flat_occupied[left] & flat_occupied[right]
    valid = candidate & flat_observed[left] & flat_observed[right]
    selected = bonds[valid]
    count = _count_by_complex(residue_left[valid], batch_id, batch_size)
    missing = _count_by_complex(residue_left[candidate & ~valid], batch_id, batch_size)
    fixed_pred = torch.where(movable[:, None, None], pred_X, given_X.detach())
    pred_atoms = fixed_pred.reshape(-1, 3)[selected]
    true_atoms = true_X.detach().reshape(-1, 3)[selected]
    _require_finite(true_atoms, 'observed reference bond coordinates')
    pred_lengths = _bond_lengths(pred_atoms)
    true_lengths = _bond_lengths(true_atoms)
    values = F.smooth_l1_loss(pred_lengths, true_lengths, reduction='none')
    return (_sum_count(values, count.sum(), pred_X), count, missing)

def paper_generator_losses(*, pred_X, true_X, true_S, movable_mask, observed_mask, batch_id, aa_feature, backbone_bonds, sequence_rounds, shadow_X=None, paratope_mask=None, distance_rounds=(), distance_pairs=None, condition_protocol='P2B', given_X=None, loss_mode='full'):
    loss_config = resolve_loss_config(loss_mode)
    active_terms = loss_config['active_terms']
    if pred_X.ndim != 3 or pred_X.shape[1:] != (14, 3) or true_X.shape != pred_X.shape:
        raise ValueError('Paper joint losses require matching [N, 14, 3] coordinates')
    protocol = condition_protocol.upper().replace('-', '')
    if protocol not in ('P1', 'P2B'):
        raise ValueError('condition_protocol must be P1 or P2-B')
    if 'bb' in active_terms and (not isinstance(backbone_bonds, torch.Tensor)):
        raise ValueError('Paper losses require backbone_bonds from the paper-contract data adapter')
    _require_finite(pred_X, 'predicted coordinates')
    movable = movable_mask.bool()
    observed = observed_mask.bool()
    if observed.shape != pred_X.shape[:2]:
        raise ValueError('Observation mask must match [N, 14] atom layout')
    n, channels = pred_X.shape[:2]
    if batch_id.shape != (n,) or movable.shape != (n,):
        raise ValueError('Residue masks and batch IDs must match N')
    batch_size = int(batch_id.max().item()) + 1 if n else 0
    occupied = aa_feature._construct_atom_pos(true_S)[:, :channels] != aa_feature.atom_pos_pad_idx
    occupied = occupied & ~aa_feature._is_global(true_S)[:, None]
    obs_occ = occupied & observed
    _require_finite(true_X[obs_occ], 'observed reference coordinates')
    if _loss_precision_risk(pred_X, true_X, obs_occ, movable, protocol, shadow_X, paratope_mask, active_terms):
        warnings.warn('Using FP64 for extreme paper-loss arithmetic; the coordinate magnitude still requires investigation.', RuntimeWarning)
        pred_X = pred_X.double()
        true_X = true_X.double()
        if given_X is not None:
            given_X = given_X.double()
        sequence_rounds = [(logits.double(), mask) for logits, mask in sequence_rounds]
        distance_rounds = [prediction.double() for prediction in distance_rounds]
        if 'shadow' in active_terms and shadow_X is not None:
            shadow_X = shadow_X.double()
    true_X = true_X.detach().to(dtype=pred_X.dtype)
    given_X = true_X if given_X is None else given_X.detach().to(dtype=pred_X.dtype)
    counts, missing = ({}, {})
    zero = _connected_zero(pred_X)
    seq_numerator = zero
    seq_counts = batch_id.new_zeros(batch_size)
    for logits, mask in sequence_rounds:
        _require_finite(logits, 'sequence logits')
        mask = mask.bool()
        seq_numerator = seq_numerator + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
        seq_counts = seq_counts + _count_by_complex(torch.nonzero(mask).flatten(), batch_id, batch_size)
    counts['sequence_residue_rounds'] = seq_counts
    seq_loss = seq_numerator / seq_counts.sum().clamp_min(1)
    eligible = torch.ones(batch_size, dtype=torch.bool, device=pred_X.device) if protocol == 'P1' else reference_coordinate_eligibility(true_X, obs_occ, movable, batch_id, batch_size)
    coordinate_mask = obs_occ & movable[:, None] & eligible[batch_id, None]
    counts['coordinate_atoms'] = _count_by_complex(torch.nonzero(coordinate_mask)[:, 0], batch_id, batch_size)
    missing['coordinate_atoms'] = _count_by_complex(torch.nonzero(occupied & ~observed & movable[:, None])[:, 0], batch_id, batch_size)
    coord_numerator = zero
    ops = []
    bb_rmsd = pred_X.new_zeros(n)
    for b in range(batch_size):
        residues = batch_id == b
        valid_atoms = coordinate_mask[residues]
        rotation = torch.eye(3, dtype=pred_X.dtype, device=pred_X.device)
        translation = pred_X.new_zeros(3)
        if bool(eligible[b]):
            if protocol != 'P1':
                alignment_mask = (obs_occ & movable[:, None])[residues, :4]
                _, rotation, translation = kabsch_torch(true_X[residues, :4][alignment_mask], pred_X[residues, :4][alignment_mask], requires_grad=True)
                _require_finite(rotation, 'alignment rotation')
                _require_finite(translation, 'alignment translation')
            aligned_valid = true_X[residues][valid_atoms] @ rotation.T + translation
            coord_numerator = coord_numerator + F.smooth_l1_loss(pred_X[residues][valid_atoms], aligned_valid, reduction='sum')
            with torch.no_grad():
                bb_valid = coordinate_mask[residues, :4]
                safe_reference = torch.where(bb_valid[..., None], true_X[residues, :4], 0)
                aligned_bb = safe_reference @ rotation.detach().T + translation.detach()
                square = ((pred_X[residues, :4] - aligned_bb) ** 2).sum(-1)
                bb_rmsd[residues] = torch.sqrt((square * bb_valid).sum(-1) / bb_valid.sum(-1).clamp_min(1))
        ops.append((rotation.detach(), translation.detach()))
    coord_loss = coord_numerator / counts['coordinate_atoms'].sum().clamp_min(1)
    bb_loss = zero
    counts['backbone_bonds'] = batch_id.new_zeros(batch_size)
    missing['backbone_bonds'] = batch_id.new_zeros(batch_size)
    if 'bb' in active_terms:
        bb_loss, counts['backbone_bonds'], missing['backbone_bonds'] = _bond_term(pred_X, true_X, given_X, movable, occupied, observed, batch_id, batch_size, backbone_bonds)
    sc_loss = zero
    counts['sidechain_bonds'] = batch_id.new_zeros(batch_size)
    missing['sidechain_bonds'] = batch_id.new_zeros(batch_size)
    if 'sc' in active_terms:
        sidechain, sidechain_mask = aa_feature.get_sidechain_bonds(true_S)
        sidechain_mask = sidechain_mask & ~aa_feature._is_global(true_S)[:, None]
        sidechain_residues = torch.nonzero(sidechain_mask)[:, 0]
        sidechain = sidechain[sidechain_mask] + sidechain_residues[:, None] * channels
        sc_loss, counts['sidechain_bonds'], missing['sidechain_bonds'] = _bond_term(pred_X, true_X, given_X, movable, occupied, observed, batch_id, batch_size, sidechain)
    shadow_loss = zero
    shadow_counts = batch_id.new_zeros(batch_size)
    missing_shadow = shadow_counts.clone()
    if 'shadow' in active_terms and shadow_X is not None:
        if paratope_mask is None:
            raise ValueError('shadow_X requires paratope_mask')
        _require_finite(shadow_X, 'shadow coordinates')
        paratope_mask = paratope_mask.bool()
        if shadow_X.shape != true_X[paratope_mask].shape:
            raise ValueError('Shadow coordinates must match selected paratope residues')
        shadow_valid = obs_occ[paratope_mask]
        paratope_ids = torch.nonzero(paratope_mask).flatten()
        shadow_counts = 3 * _count_by_complex(paratope_ids[torch.nonzero(shadow_valid)[:, 0]], batch_id, batch_size)
        missing_shadow = 3 * _count_by_complex(paratope_ids[torch.nonzero((occupied & ~observed)[paratope_mask])[:, 0]], batch_id, batch_size)
        shadow_values = F.smooth_l1_loss(shadow_X[shadow_valid], true_X[paratope_mask][shadow_valid], reduction='none')
        shadow_loss = _sum_count(shadow_values, shadow_counts.sum(), shadow_X)
    counts['shadow_scalar_coordinates'] = shadow_counts
    missing['shadow_scalar_coordinates'] = missing_shadow
    distance_losses = []
    distance_rounds = list(distance_rounds)
    if distance_rounds:
        if distance_pairs is None or distance_pairs.ndim != 2 or distance_pairs.shape[0] != 2:
            raise ValueError('Distance predictions require [2, E] global residue pairs')
        left, right = distance_pairs
        if distance_pairs.numel() and (bool((distance_pairs < 0).any()) or bool((distance_pairs >= n).any())):
            raise ValueError('Distance pair residue index out of range')
        if bool((batch_id[left] != batch_id[right]).any()):
            raise ValueError('Distance pairs must not cross complexes')
        complete = (~occupied | observed).all(-1) & occupied.any(-1)
        valid_pairs = complete[left] & complete[right]
        pair_count = _count_by_complex(left[valid_pairs], batch_id, batch_size)
        missing_pair = _count_by_complex(left[~valid_pairs], batch_id, batch_size)
        eligible_left, eligible_right = (left[valid_pairs], right[valid_pairs])
        with torch.no_grad():
            differences = true_X[eligible_left, :, None] - true_X[eligible_right, None, :]
            distances = torch.linalg.vector_norm(differences, dim=-1)
            canonical_pairs = occupied[eligible_left, :, None] & occupied[eligible_right, None, :]
            target_distances = distances.masked_fill(~canonical_pairs, float('inf')).flatten(1).min(-1).values / 10
        for t, prediction in enumerate(distance_rounds):
            prediction = prediction.reshape(-1)
            if prediction.shape != left.shape:
                raise ValueError('Distance prediction must match candidate pair order')
            _require_finite(prediction, 'pair distance predictions')
            values = F.smooth_l1_loss(prediction[valid_pairs], target_distances, reduction='none')
            distance_losses.append(_sum_count(values, pair_count.sum(), prediction))
            counts['distance_pairs_round_%d' % t] = pair_count
            missing['distance_pairs_round_%d' % t] = missing_pair
    distance_loss = sum(distance_losses, zero)

    def package(values):
        return {'per_batch': {k: v.sum().detach() for k, v in values.items()}, 'per_complex': {k: v.detach() for k, v in values.items()}}
    terms = {'seq': seq_loss, 'coord': coord_loss, 'bb': bb_loss, 'sc': sc_loss, 'shadow': shadow_loss, 'dist': distance_loss}
    total = terms[active_terms[0]]
    for name in active_terms[1:]:
        total = total + terms[name]
    return {**terms, 'total': total, 'loss_config': loss_config, 'distance_round_losses': distance_losses, 'counts': package(counts), 'missing_observations': package(missing), 'coordinate_eligible': eligible, 'ops': ops, 'bb_rmsd': bb_rmsd[movable]}
