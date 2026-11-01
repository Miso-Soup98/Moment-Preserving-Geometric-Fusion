from dataclasses import dataclass
import torch
from torch import nn
from .fusion import MomentGeometricFusion
from .schemas import PreparedMoments
SUPPORTED_VARIANTS = ('mpgf',)

def canonical_mapping(vocab):
    symbols = set('ACDEFGHIKLMNPQRSTVWY')
    ids = [i for i in range(len(vocab)) if vocab.idx_to_symbol(i) in symbols]
    if len(ids) != 20 or len({vocab.idx_to_symbol(i) for i in ids}) != 20:
        raise ValueError('The backbone vocabulary must contain exactly the 20 standard identities')
    return ids

def weighted_role_table(feature, canonical_ids):
    ids = torch.as_tensor(canonical_ids, device=feature.residue_atom_type.device)
    types = feature.residue_atom_type[ids]
    positions = feature.residue_atom_pos[ids]
    roles = feature.aa_embedding.atom_embedding(types) + feature.aa_embedding.atom_pos_embedding(positions)
    return roles * feature.get_atom_weights(ids).unsqueeze(-1)

def install_fusion(model, variant, config=None, paper_contract=False, layer_type=None):
    from data.pdb_utils import VOCAB
    layers = []
    if layer_type is None:
        layers = [m for m in model.modules() if getattr(type(m), '_mpgf_owned_message_layer', False)]
        if not layers:
            from models.mpgf.layers import MPGFMessageLayer
            layer_type = MPGFMessageLayer
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError('Unsupported fusion variant: {0}'.format(variant))
    resolved = dict(epsilon=0.0001, distance_epsilon=0.0001, length_scale=1.0, node_chunk_size=256, edge_chunk_size=512)
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(resolved)
    if unknown:
        raise ValueError('Unknown fusion configuration fields: {0}'.format(sorted(unknown)))
    resolved.update(supplied)
    if resolved['length_scale'] != 1.0:
        raise ValueError('Model coordinates are normalized by 10 Angstrom; adapter length_scale must be 1')
    ids = canonical_mapping(VOCAB)
    if ids != list(range(model.num_classes)):
        raise ValueError('Existing sequence-head class mapping differs from standard token mapping')
    feature = model.aa_feature
    c = feature.residue_atom_type.shape[1]
    d = feature.aa_embedding.atom_embedding.embedding_dim
    if layer_type is not None:
        layers = [m for m in model.modules() if isinstance(m, layer_type)]
    core_variant = 'mpgf'
    for layer in layers:
        layer.fusion = MomentGeometricFusion(c, d, layer.radial_linear.out_features, variant=core_variant, paper_contract=paper_contract, **resolved)
    metadata = dict(variant=variant, core_variant=core_variant, contract_version='MPGF-v1', canonical_token_ids=ids, residue_symbols=[VOCAB.idx_to_symbol(i) for i in ids], ordered_atom_names=[VOCAB.backbone_atoms + VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i)) for i in ids], atom_slots=c, role_dim=d, role_scale_mode='embedding', coordinate_unit='Angstrom', coordinate_divisor=10.0, backbone_role_note='CA and C share carbon plus backbone embedding', probability_source='uniform at round 0; previous sequence-head softmax thereafter', config=resolved)
    if paper_contract:
        metadata.update(paper_contract='MPGF-v1', global_reader='deterministic [B_g,0] with shared lifted reader', full_distance_occupancy='canonical chemical occupancy; MASK and global all slots', projection='affine, column-major vectorization')
    return metadata

@dataclass
class GeometryEdgePlan:
    auxiliary_edges: torch.Tensor
    auxiliary_positions: torch.Tensor
    real_edges: torch.Tensor
    real_positions: torch.Tensor
    edge_count: int
    source_tensors: tuple

@dataclass
class FusionContext:
    prob: torch.Tensor
    roles: torch.Tensor
    real_mask: torch.Tensor
    design_mask: torch.Tensor
    batch_index: torch.Tensor
    source_index: torch.Tensor
    prepared: object = None
    paper_contract: bool = False
    chemical_occupancy: object = None

    def __post_init__(self):
        self.real_indices = torch.nonzero(self.real_mask, as_tuple=False).flatten()
        self.compact = torch.full_like(self.source_index, -1)
        self.reader_indices = torch.arange(len(self.real_mask), device=self.prob.device) if self.paper_contract else self.real_indices
        self.compact[self.reader_indices] = torch.arange(self.reader_indices.numel(), device=self.prob.device)
        self.real_prob = self.prob.index_select(0, self.real_indices)
        self.real_design_mask = self.design_mask.index_select(0, self.real_indices)
        self.real_batch_index = self.batch_index.index_select(0, self.real_indices)
        self._edge_plans = {}

    def edge_plan(self, edge_index):
        sources = (edge_index,) if isinstance(edge_index, torch.Tensor) else tuple(edge_index)
        cache_key = None
        if all((isinstance(t, torch.Tensor) for t in sources)):
            try:
                cache_key = tuple(((id(t), t._version) for t in sources))
            except RuntimeError:
                pass
        if cache_key is not None and cache_key in self._edge_plans:
            return self._edge_plans[cache_key]
        edges = edge_index if isinstance(edge_index, torch.Tensor) else torch.stack(sources)
        if edges.dtype != torch.long:
            raise TypeError('edge_index must be torch.long')
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.device != self.prob.device:
            raise ValueError('edge_index must have shape [2,E] on input device')
        if edges.numel() and bool(((edges < 0) | (edges >= len(self.real_mask))).any()):
            raise ValueError('edge_index outside node range')
        row, col = edges
        if bool((self.batch_index[row] != self.batch_index[col]).any()):
            raise ValueError('Cross-batch edges are forbidden')
        use_new = torch.ones_like(row, dtype=torch.bool) if self.paper_contract else self.real_mask[row] & self.real_mask[col]
        real_positions = torch.nonzero(use_new, as_tuple=False).flatten()
        auxiliary_positions = torch.nonzero(~use_new, as_tuple=False).flatten()
        plan = GeometryEdgePlan(edges.index_select(1, auxiliary_positions), auxiliary_positions, self.compact[edges.index_select(1, real_positions)], real_positions, edges.shape[1], sources)
        if cache_key is not None:
            self._edge_plans[cache_key] = plan
        return plan

    def subset(self, mask):
        idx = torch.arange(len(self.real_mask), device=self.real_mask.device)[mask]
        real_idx = self.compact[idx] if self.paper_contract else self.compact[idx[self.real_mask[idx]]]
        prepared = None if self.prepared is None else self.prepared.index_select(real_idx)
        return FusionContext(self.prob[idx], self.roles, self.real_mask[idx], self.design_mask[idx], self.batch_index[idx], self.source_index[idx], prepared, self.paper_contract, None if self.chemical_occupancy is None else self.chemical_occupancy[idx])

def build_fusion_context(model, tokens, smooth_prob, smooth_mask, batch_index):
    from data.pdb_utils import VOCAB
    ids = model.fusion_metadata['canonical_token_ids']
    feature = model.aa_feature
    if bool((tokens == VOCAB.get_pad_idx()).any()):
        raise ValueError('PAD nodes must not enter the MPGF graph')
    real = ~feature._is_global(tokens)
    unknown = tokens == feature.mask_idx
    known = real & ~unknown
    lut = torch.full((len(VOCAB),), -1, dtype=torch.long, device=tokens.device)
    lut[torch.tensor(ids, device=tokens.device)] = torch.arange(len(ids), device=tokens.device)
    if bool(((tokens < 0) | (tokens >= len(VOCAB))).any()):
        raise ValueError('Token index outside vocabulary')
    if bool((lut[tokens[known]] < 0).any()):
        raise ValueError('Unsupported residue identity')
    roles = weighted_role_table(feature, ids)
    prob = roles.new_zeros((len(tokens), len(ids)))
    prob[known] = torch.nn.functional.one_hot(lut[tokens[known]], len(ids)).to(roles.dtype)
    prob[unknown] = 1.0 / len(ids)
    if smooth_mask is None:
        smooth_mask = unknown
    if smooth_mask.dtype != torch.bool or smooth_mask.shape != tokens.shape:
        raise ValueError('smooth_mask must be bool [N]')
    if bool((smooth_mask & ~real).any()):
        raise ValueError('Special nodes cannot be design residues')
    if smooth_prob is not None:
        if smooth_prob.shape != (int(smooth_mask.sum()), len(ids)):
            raise ValueError('Previous sequence probabilities must align with smooth_mask')
        prob[smooth_mask] = smooth_prob.to(roles.dtype)
    first = next((m.fusion for m in model.modules() if hasattr(m, 'fusion')))
    paper_contract = getattr(first, 'paper_contract', False)
    occupancy = None
    if paper_contract:
        occupancy = torch.ones((len(tokens), roles.shape[1]), dtype=torch.bool, device=tokens.device)
        occupancy[known] = feature.residue_atom_type[tokens[known]] != feature.atom_pad_idx
    context = FusionContext(prob, roles, real, unknown, batch_index, torch.arange(len(tokens), device=tokens.device), paper_contract=paper_contract, chemical_occupancy=occupancy)
    anon = None
    real_prepared = first.prepare(context.real_prob, roles, design_mask=context.real_design_mask, anonymous_roles=anon, chemical_occupancy=None if occupancy is None else occupancy[real])
    if paper_contract:
        global_tokens = tokens[~real]
        global_roles = feature.aa_embedding.atom_embedding(feature.residue_atom_type[global_tokens]) + feature.aa_embedding.atom_pos_embedding(feature.residue_atom_pos[global_tokens])
        global_roles = global_roles * feature.get_atom_weights(global_tokens).unsqueeze(-1)
        context.prepared = _with_deterministic_globals(real_prepared, real, global_roles, occupancy)
    else:
        context.prepared = real_prepared
    return context

def _with_deterministic_globals(real_prepared, real_mask, global_roles, occupancy):
    real_indices = torch.nonzero(real_mask, as_tuple=False).flatten()
    global_indices = torch.nonzero(~real_mask, as_tuple=False).flatten()
    c, d = real_prepared.mean.shape[1:]
    mean = real_prepared.mean.new_zeros((len(real_mask), c, d))
    mean = mean.index_copy(0, real_indices, real_prepared.mean)
    mean = mean.index_copy(0, global_indices, global_roles.to(mean.dtype))
    covariance = real_prepared.row_covariance.new_zeros((len(real_mask), c, c))
    covariance = covariance.index_copy(0, real_indices, real_prepared.row_covariance)
    lift = real_prepared.lift.new_zeros((len(real_mask), c, c + d))
    lift = lift.index_copy(0, real_indices, real_prepared.lift)
    global_lift = torch.cat((global_roles.to(lift.dtype), lift.new_zeros((len(global_indices), c, c))), -1)
    lift = lift.index_copy(0, global_indices, global_lift)
    return PreparedMoments(mean, covariance, lift, real_prepared.variant, occupancy)

def dispatch_geometry(fusion, context, edge_index, coords, attrs, weights, auxiliary_radial, *, plan=None):
    plan = context.edge_plan(edge_index) if plan is None else plan
    if auxiliary_radial.shape[0] == plan.edge_count:
        old_features = auxiliary_radial.index_select(0, plan.auxiliary_positions)
    elif auxiliary_radial.shape[0] == plan.auxiliary_positions.numel():
        old_features = auxiliary_radial
    else:
        raise ValueError('Auxiliary radial rows must match all edges or special edges')
    real_coords = coords.index_select(0, context.reader_indices)
    if context.prepared is None and context.paper_contract:
        anon = (attrs * weights.unsqueeze(-1)).index_select(0, context.real_indices)
        prepared = fusion.prepare(context.real_prob, context.roles, design_mask=context.real_design_mask, anonymous_roles=anon, chemical_occupancy=None if context.chemical_occupancy is None else context.chemical_occupancy[context.real_mask])
        prepared = _with_deterministic_globals(prepared, context.real_mask, (attrs * weights.unsqueeze(-1))[~context.real_mask], context.chemical_occupancy)
        output = fusion._forward_prepared(real_coords, prepared, plan.real_edges)
    elif context.prepared is None:
        anon = (attrs * weights.unsqueeze(-1)).index_select(0, context.real_indices)
        output = fusion(real_coords, context.real_prob, context.roles, plan.real_edges, design_mask=context.real_design_mask, batch_index=context.real_batch_index, anonymous_roles=anon)
    else:
        output = fusion._forward_prepared(real_coords, context.prepared, plan.real_edges)
    result = auxiliary_radial.new_zeros((plan.edge_count, auxiliary_radial.shape[-1]))
    result.index_copy_(0, plan.auxiliary_positions, old_features)
    result.index_copy_(0, plan.real_positions, output.edge_features.to(result.dtype))
    return result + auxiliary_radial.sum() * 0 + output.edge_features.sum() * 0
