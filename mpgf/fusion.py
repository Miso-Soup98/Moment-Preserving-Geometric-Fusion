from __future__ import annotations
from dataclasses import asdict
import math
import torch
from torch import nn
from .moments import _role_moments_unchecked, validate_roles, check_floating, computation_dtype
from .matrix_functions import null_preserving_sqrt
from .schemas import FusionConfig, FusionOutput, PreparedMoments

def bounded_readout(raw):
    check_floating(raw, 'raw_relation')
    if raw.ndim < 2:
        raise ValueError('raw_relation requires two matrix dimensions')
    with torch.autocast(device_type=raw.device.type, enabled=False):
        return _bounded_readout_unchecked(raw)

def _bounded_readout_unchecked(raw):
    work = raw.to(computation_dtype(raw))
    scale = work.abs().amax((-2, -1), keepdim=True).clamp_min(1)
    scaled = work / scale
    denominator = torch.sqrt(scale.reciprocal().square() + scaled.square().sum((-2, -1), keepdim=True))
    return scaled / denominator

def _validate_distance_scales(dtype, length_scale, distance_epsilon):
    if not math.isfinite(length_scale) or length_scale <= 0 or (not math.isfinite(distance_epsilon)) or (distance_epsilon <= 0):
        raise ValueError('Distance scale and epsilon must be finite and positive')
    limits = torch.finfo(dtype)
    if not limits.tiny <= length_scale <= limits.max:
        raise ValueError('length_scale is not representable in the computation dtype')
    if not math.sqrt(limits.tiny) <= distance_epsilon <= math.sqrt(limits.max):
        raise ValueError('distance_epsilon squared is not representable in the computation dtype')

def _distance_norm(delta, distance_epsilon):
    norm = torch.hypot(delta[..., 0], delta.new_tensor(distance_epsilon))
    norm = torch.hypot(norm, delta[..., 1])
    norm = torch.hypot(norm, delta[..., 2])
    return norm

class _SmoothDistance(torch.autograd.Function):

    @staticmethod
    def forward(ctx, source, target, length_scale, distance_epsilon):
        delta = (source.unsqueeze(-2) - target.unsqueeze(-3)) / length_scale
        norm = _distance_norm(delta, distance_epsilon)
        ctx.save_for_backward(source, target, norm)
        ctx.length_scale, ctx.distance_epsilon = (length_scale, distance_epsilon)
        return norm - distance_epsilon

    @staticmethod
    def backward(ctx, grad_output):
        source, target, norm = ctx.saved_tensors
        delta = (source.unsqueeze(-2) - target.unsqueeze(-3)) / ctx.length_scale
        if torch.is_grad_enabled():
            norm = _distance_norm(delta, ctx.distance_epsilon)
        direction = delta / norm.unsqueeze(-1) / ctx.length_scale
        gradient = grad_output.unsqueeze(-1) * direction
        return (gradient.sum(-2), -gradient.sum(-3), None, None)

def _smooth_distances_unchecked(source_coords, target_coords, length_scale, distance_epsilon, dtype):
    return _SmoothDistance.apply(source_coords.to(dtype), target_coords.to(dtype), length_scale, distance_epsilon)

def smooth_distances(source_coords, target_coords, length_scale=1.0, distance_epsilon=0.0001):
    check_floating(source_coords, 'source_coords')
    check_floating(target_coords, 'target_coords')
    if source_coords.shape[:-2] != target_coords.shape[:-2] or source_coords.shape[-1] != 3 or target_coords.shape[-1] != 3:
        raise ValueError('Coordinates must have matching leading dimensions and xyz')
    if source_coords.device != target_coords.device:
        raise ValueError('Coordinate devices differ')
    dtype = computation_dtype(source_coords, target_coords)
    _validate_distance_scales(dtype, length_scale, distance_epsilon)
    with torch.autocast(device_type=source_coords.device.type, enabled=False):
        return _smooth_distances_unchecked(source_coords, target_coords, length_scale, distance_epsilon, dtype)

class MomentGeometricFusion(nn.Module):

    def __init__(self, atom_slots, role_dim, output_dim, variant='mpgf', epsilon=0.0001, distance_epsilon=0.0001, length_scale=1.0, node_chunk_size=256, edge_chunk_size=512, alpha=1.0, role_scale_mode='embedding', paper_contract=False):
        super().__init__()
        self.config = FusionConfig(atom_slots, role_dim, output_dim, variant, epsilon, distance_epsilon, length_scale, node_chunk_size, edge_chunk_size, alpha, role_scale_mode)
        if not isinstance(paper_contract, bool):
            raise TypeError('paper_contract must be bool')
        self.paper_contract = paper_contract
        self.relation_width = atom_slots + role_dim
        self.projection = nn.Linear(self.relation_width ** 2, output_dim, bias=paper_contract)

    def get_extra_state(self):
        state = asdict(self.config)
        if getattr(self, 'paper_contract', False):
            state['paper_contract'] = 'MPGF-v1'
        return state

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if state != expected:
            diff = [k for k in set(expected) | set(state) if state.get(k) != expected.get(k)]
            raise RuntimeError('Fusion checkpoint configuration mismatch: %s' % ', '.join(diff))

    def prepare(self, identity_prob, role_table, design_mask=None, anonymous_roles=None, chemical_occupancy=None):
        validate_roles(identity_prob, role_table)
        return self._prepare_validated(identity_prob, role_table, design_mask, anonymous_roles, chemical_occupancy)

    def _prepare_validated(self, identity_prob, role_table, design_mask, anonymous_roles, chemical_occupancy=None):
        cfg = self.config
        if role_table.shape[1:] != (cfg.atom_slots, cfg.role_dim):
            raise ValueError('role_table dimensions disagree with configured C,d')
        prob = identity_prob
        occupancy = None
        with torch.autocast(device_type=role_table.device.type, enabled=False):
            mean, cov = _role_moments_unchecked(prob, role_table, cfg.node_chunk_size)
            root = null_preserving_sqrt(cov, cfg.epsilon)
            root = root * math.sqrt(cfg.alpha)
            lift = torch.cat((mean, root), dim=-1)
            return PreparedMoments(mean, cov, lift, cfg.variant, occupancy)

    def forward(self, coords, identity_prob, role_table, edge_index, *, return_diagnostics=False, design_mask=None, batch_index=None, prepared=None, anonymous_roles=None, chemical_occupancy=None):
        cfg = self.config
        check_floating(coords, 'coords')
        validate_roles(identity_prob, role_table)
        if coords.ndim != 3 or coords.shape[1:] != (cfg.atom_slots, 3) or len(coords) != len(identity_prob):
            raise ValueError('coords must have shape [N,C,3] matching probabilities and configured slots')
        if coords.device != identity_prob.device or coords.device != self.projection.weight.device:
            raise ValueError('coords, probabilities, roles and module must share a device')
        if role_table.shape[1:] != (cfg.atom_slots, cfg.role_dim):
            raise ValueError('role_table dimensions disagree with configured C,d')
        if not isinstance(edge_index, torch.Tensor) or edge_index.dtype != torch.long:
            raise TypeError('edge_index must be torch.long')
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.device != coords.device:
            raise ValueError('edge_index must have shape [2,E] on input device')
        if edge_index.numel() and (bool((edge_index < 0).any()) or bool((edge_index >= len(coords)).any())):
            raise ValueError('edge_index outside real residue range')
        if batch_index is not None:
            if batch_index.shape != (len(coords),) or batch_index.dtype != torch.long or batch_index.device != coords.device:
                raise ValueError('batch_index must be long [N] on input device')
            if bool((batch_index[edge_index[0]] != batch_index[edge_index[1]]).any()):
                raise ValueError('Cross-batch edges are forbidden')
        prepared = self._prepare_validated(identity_prob, role_table, design_mask, anonymous_roles, chemical_occupancy) if prepared is None else prepared
        return self._execute_prepared(coords, prepared, edge_index, return_diagnostics=return_diagnostics)

    def _forward_prepared(self, coords, prepared, edge_index, *, return_diagnostics=False):
        cfg = self.config
        check_floating(coords, 'coords')
        if coords.ndim != 3 or coords.shape[1:] != (cfg.atom_slots, 3):
            raise ValueError('coords must have shape [N,C,3] matching configured slots')
        if coords.device != self.projection.weight.device:
            raise ValueError('coords and module must share a device')
        if not isinstance(edge_index, torch.Tensor) or edge_index.dtype != torch.long:
            raise TypeError('edge_index must be torch.long')
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.device != coords.device:
            raise ValueError('edge_index must have shape [2,E] on input device')
        return self._execute_prepared(coords, prepared, edge_index, return_diagnostics=return_diagnostics)

    def _execute_prepared(self, coords, prepared, edge_index, *, return_diagnostics):
        cfg = self.config
        if prepared.lift.shape != (len(coords), cfg.atom_slots, cfg.atom_slots + cfg.role_dim) or prepared.lift.device != coords.device:
            raise ValueError('Prepared moments do not match node count, C,d or device')
        if prepared.variant != cfg.variant:
            raise ValueError('Prepared moments belong to a different fusion variant')
        occupancy = getattr(prepared, 'chemical_occupancy', None)
        distance_dtype = computation_dtype(coords)
        _validate_distance_scales(distance_dtype, cfg.length_scale, cfg.distance_epsilon)
        lift = prepared.lift.to(computation_dtype(coords, prepared.lift))
        features = []
        raws = []
        normalized = []
        chunk_raw_finite = []
        edge_count = edge_index.shape[1]
        all_raw_finite = torch.ones((), dtype=torch.bool, device=coords.device)
        for start in range(0, max(edge_count, 1), cfg.edge_chunk_size):
            row, col = edge_index[:, start:start + cfg.edge_chunk_size]
            with torch.autocast(device_type=coords.device.type, enabled=False):
                distance = _smooth_distances_unchecked(coords[row], coords[col], cfg.length_scale, cfg.distance_epsilon, distance_dtype)
                raw = lift[row].transpose(-1, -2) @ distance.to(lift.dtype) @ lift[col]
                finite = torch.isfinite(raw).all()
                chunk_raw_finite.append(finite)
                all_raw_finite = all_raw_finite & finite
                norm = _bounded_readout_unchecked(raw)
            flattened = norm.transpose(-1, -2).flatten(1) if getattr(self, 'paper_contract', False) else norm.flatten(1)
            feature = self.projection(flattened.to(self.projection.weight.dtype))
            features.append(feature)
            if return_diagnostics:
                raws.append(raw)
                normalized.append(norm)
        if not bool(all_raw_finite):
            if distance_dtype == torch.float64 and lift.dtype == torch.float64:
                raise ValueError('raw_relation contains nonfinite values')
            bad_chunks = torch.nonzero(~torch.stack(chunk_raw_finite), as_tuple=False).flatten().cpu().tolist()
            for chunk in bad_chunks:
                start = chunk * cfg.edge_chunk_size
                row, col = edge_index[:, start:start + cfg.edge_chunk_size]
                with torch.autocast(device_type=coords.device.type, enabled=False):
                    distance = _smooth_distances_unchecked(coords[row], coords[col], cfg.length_scale, cfg.distance_epsilon, torch.float64)
                    raw = lift[row].double().transpose(-1, -2) @ distance @ lift[col].double()
                    check_floating(raw, 'raw_relation')
                    norm = _bounded_readout_unchecked(raw)
                flattened = norm.transpose(-1, -2).flatten(1) if getattr(self, 'paper_contract', False) else norm.flatten(1)
                features[chunk] = self.projection(flattened.to(self.projection.weight.dtype))
                if return_diagnostics:
                    raws[chunk] = raw
                    normalized[chunk] = norm
        return FusionOutput(torch.cat(features), torch.cat(raws) if return_diagnostics else None, torch.cat(normalized) if return_diagnostics else None, prepared.mean if return_diagnostics else None, prepared.row_covariance if return_diagnostics else None)
