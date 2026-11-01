import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from data.pdb_utils import VOCAB
from .mpgf.features import MPGFSeparatedAminoAcidFeature, MPGFProteinFeature, MPGFGraphEdgeConstructor, MPGFCoordinateNormalizer, _knn_edges
from evaluation.rmsd import kabsch_torch
from utils.loss_config import resolve_loss_config
from .mpgf.encoder import MPGFEncoder
from .mpgf.layers import MPGFGraphNetwork, MPGFMessageLayer

class MPGFModel(nn.Module):

    def __init__(self, embed_size, hidden_size, n_channel, num_classes, mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6, n_layers=3, iter_round=3, dropout=0.1, struct_only=False, backbone_only=False, fix_channel_weights=False, pred_edge_dist=True, keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False, fusion_variant='mpgf', fusion_config=None, paper_contract=True, condition='P2-B', loss_mode='full') -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.struct_only = struct_only
        self.paper_contract = bool(paper_contract)
        resolved_loss = resolve_loss_config(loss_mode)
        self.seq_loss_weight = self.seq_contact_weight = 1.0
        if not self.paper_contract and loss_mode != 'full':
            raise ValueError('Reduced loss modes require the MPGF paper contract')
        self.loss_mode = loss_mode
        if self.paper_contract:
            self.loss_config = resolved_loss
        self.condition_protocol = condition
        if condition not in ('P2-B', 'P1'):
            raise ValueError('condition must be P2-B or P1')
        if condition == 'P1' and (not self.paper_contract):
            raise ValueError('P1 requires the paper condition adapter')
        if self.paper_contract and (struct_only or backbone_only or n_channel != 14):
            raise ValueError('Paper MPGF implements joint sequence-structure generation with all 14 slots')
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        if self.backbone_only:
            n_channel = 4
        self.cdr_type = cdr_type
        self.paratope = paratope
        atom_embed_size = embed_size // 4
        self.aa_feature = MPGFSeparatedAminoAcidFeature(embed_size, atom_embed_size, relative_position=relative_position, edge_constructor=MPGFGraphEdgeConstructor, fix_atom_weights=fix_channel_weights, backbone_only=backbone_only)
        self.protein_feature = MPGFProteinFeature(backbone_only=backbone_only)
        if keep_memory:
            self.memory_ffn = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, embed_size))
        if self.pred_edge_dist:
            if self.keep_memory:
                self.edge_H_ffn = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            self.edge_dist_ffn = nn.Sequential(nn.SiLU(), nn.Linear(2 * hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))
            self.init_gnn = MPGFGraphNetwork(embed_size, hidden_size, hidden_size, n_channel, channel_nf=atom_embed_size, radial_nf=hidden_size, in_edge_nf=0, n_layers=n_layers, residual=True, dropout=dropout, dense=False)
        if not struct_only:
            self.ffn_residue = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, self.num_classes))
        else:
            self.prmsd_ffn = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))
        self.gnn = MPGFEncoder(embed_size, hidden_size, hidden_size, n_channel, channel_nf=atom_embed_size, radial_nf=hidden_size, in_edge_nf=0, n_layers=n_layers, residual=True, dropout=dropout, dense=False)
        self.normalizer = MPGFCoordinateNormalizer()
        self.fusion_variant = fusion_variant
        self.fusion_metadata = {'variant': fusion_variant}
        from mpgf.adapters import install_fusion
        self.fusion_metadata = install_fusion(self, fusion_variant, fusion_config, paper_contract=self.paper_contract, layer_type=MPGFMessageLayer)
        if self.paper_contract:
            for module in self.modules():
                if isinstance(module, MPGFMessageLayer):
                    module.paper_contract = True
            self.fusion_metadata = dict(self.fusion_metadata, paper_contract='MPGF-v1', condition_protocol=condition)
        self.batch_constants = {}

    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return (X, S)

    def message_passing(self, X, S, residue_pos, interface_X, paratope_mask, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None):
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)
        fusion_context = None
        from mpgf.adapters import build_fusion_context
        fusion_context = build_fusion_context(self, S, smooth_prob, smooth_mask, batch_id)
        if not self.keep_memory:
            memory_H = None
        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)
        coordinate_callback = self.batch_constants.get('coordinate_callback')
        if self.pred_edge_dist:
            if memory_H is not None:
                edge_H = self.edge_H_ffn(memory_H)
            else:
                edge_H, dumb_X = self.init_gnn(H_0, X, ctx_edges, channel_attr=atom_embeddings, channel_weights=atom_weights, fusion_context=fusion_context, coordinate_callback=coordinate_callback)
                X = X + dumb_X * 0
        X = self.aa_feature.update_globel_coordinates(X, S)
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        local_ctx_edges = self.batch_constants['local_ctx_edges']
        local_inter_edges = self.batch_constants['local_inter_edges']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        local_ctx_edges = _knn_edges(local_X, atom_pos, local_ctx_edges.T, self.aa_feature.atom_pos_pad_idx, self.k_neighbors, (offsets, local_batch_id, max_n, gni2lni))
        local_X[local_is_ab] = interface_X
        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src_H, dst_H = (local_H[local_inter_edges[0]], local_H[local_inter_edges[1]])
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) + self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))
            p_edge_dist = p_edge_dist.squeeze()
        else:
            p_edge_dist = None
        local_inter_edges = _knn_edges(local_X, atom_pos, local_inter_edges.T, self.aa_feature.atom_pos_pad_idx, self.k_neighbors, (offsets, local_batch_id, max_n, gni2lni), given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)
        inter_coordinate_callback = None
        if coordinate_callback is not None:
            fixed_local_X = X[local_mask].detach()

            def inter_coordinate_callback(current):
                return torch.where(local_is_ab[:, None, None], current, fixed_local_X)
        H, pred_X, pred_local_X = self.gnn(H_0, X, ctx_edges, local_mask, local_X, local_edges, paratope_mask, local_is_ab, channel_attr=atom_embeddings, channel_weights=atom_weights, fusion_context=fusion_context, coordinate_callback=coordinate_callback, inter_coordinate_callback=inter_coordinate_callback)
        interface_X = pred_local_X[local_is_ab]
        pred_logits = None if self.struct_only else self.ffn_residue(H)
        return (pred_logits, pred_X, interface_X, H, p_edge_dist)

    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        if getattr(self, 'paper_contract', False) and getattr(self, 'condition_protocol', 'P2-B') == 'P1':
            if init_noise is not None:
                raise ValueError('P1 has no sampling-noise input')
            return X[paratope_mask].clone()
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10 + ca_noise.unsqueeze(1)
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise
        return init_local_X

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        batch_id = torch.zeros_like(S)
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1
        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(paratope_mask, torch.logical_and(is_ag, not_ag_global))
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = (local_segment_ids[row], local_segment_ids[col])
        is_ctx = row_segment_ids == col_segment_ids
        is_inter = torch.logical_not(is_ctx)
        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)
        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id

    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)
        dist = torch.norm(dist, dim=-1)
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))
        dist = dist + pos_pad * 10000000000.0
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]
        return dist
        is_binding = dist <= self.bind_dist_cutoff
        return is_binding

    def _fusion_prob_snapshot(self, S, prob, smask):
        snapshot = self.aa_feature.atom_weight.new_zeros((len(S), self.num_classes))
        real = ~self.aa_feature._is_global(S)
        known = real & (S != self.mask_id)
        snapshot[known] = F.one_hot(S[known], num_classes=self.num_classes).to(snapshot.dtype)
        snapshot[real & ~known] = 1.0 / self.num_classes
        if prob is not None and bool(smask.any()):
            snapshot[smask] = prob
        return snapshot
        snapshot = F.one_hot(S.clamp(min=0, max=self.num_classes - 1), num_classes=self.num_classes).float()
        if prob is not None and bool(smask.any()):
            snapshot[smask] = prob
        return snapshot

    def _forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None, return_traj=False):
        batch_id = self.batch_constants['batch_id']
        X, S = self.init_mask(X, S, cmask, smask, template)
        is_p1 = getattr(self, 'paper_contract', False) and getattr(self, 'condition_protocol', 'P2-B') == 'P1'
        if is_p1:
            supplied_X = X.detach().clone()
            if not torch.equal(cmask, paratope_mask) or bool((smask & ~cmask).any()):
                raise ValueError('P1 coordinate and paratope masks must be the same H3 region')
            X = self.aa_feature.update_globel_coordinates(X, S)
            centers = X[S == self.aa_feature.boa_idx][:, 0].detach()
            self.batch_constants['p1_centers'] = centers
            X = self.normalizer.normalize(X - centers[batch_id, None])
            fixed_X = X.detach().clone()
            fixed_mask = ~cmask & ~self.aa_feature._is_global(S)

            def clamp_coordinates(current):
                current = torch.where(fixed_mask[:, None, None], fixed_X, current)
                return self.aa_feature.update_globel_coordinates(current, S)
            self.batch_constants['coordinate_callback'] = clamp_coordinates
        else:
            X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
            X = self.normalizer.normalize(X)
        X = self.aa_feature.update_globel_coordinates(X, S)
        interface_X = self.init_interface(X, S, paratope_mask, batch_id, init_noise)
        r_pred_S_logits, pred_S_dist = ([], None)
        r_interface_X = [interface_X.clone()]
        r_edge_dist = []
        r_prob_traj, r_coord_traj = ([], [])
        if return_traj:
            uniform_prob = torch.full((smask.sum(), self.num_classes), 1.0 / self.num_classes, device=X.device)
            r_prob_traj.append(self._fusion_prob_snapshot(S, uniform_prob, smask))
            r_coord_traj.append(X.clone())
        memory_H = None
        for t in range(self.round):
            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(X, S, residue_pos, interface_X, paratope_mask, batch_id, t, memory_H, pred_S_dist, smask)
            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_edge_dist.append(edge_dist)
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_globel_coordinates(X, S)
            if not self.struct_only:
                S = S.clone()
                if t == self.round - 1:
                    S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask], dim=-1)
            if return_traj:
                snapshot_prob = pred_S_dist
                if not self.struct_only:
                    snapshot_prob = torch.softmax(pred_S_logits[smask], dim=-1)
                r_prob_traj.append(self._fusion_prob_snapshot(S, snapshot_prob, smask))
                r_coord_traj.append(X.clone())
        interface_batch_id = self.batch_constants['interface_batch_id']
        if self.struct_only:
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()
        else:
            prmsd = None
        pred_X = self.normalizer.unnormalize(pred_X)
        if is_p1:
            pred_X = pred_X + centers[batch_id, None]
            pred_X = torch.where(fixed_mask[:, None, None], supplied_X, pred_X)
            pred_X = self.aa_feature.update_globel_coordinates(pred_X, S)
        else:
            pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X in enumerate(r_interface_X):
            interface_X = self.normalizer.unnormalize(interface_X)
            interface_X = interface_X + centers[interface_batch_id, None] if is_p1 else self.normalizer.uncentering(interface_X, interface_batch_id, _type=4)
            r_interface_X[i] = interface_X
        if return_traj:
            for i, coord_X in enumerate(r_coord_traj):
                coord_X = self.normalizer.unnormalize(coord_X)
                coord_X = coord_X + centers[batch_id, None] if is_p1 else self.normalizer.uncentering(coord_X, batch_id)
                r_coord_traj[i] = coord_X
        self.normalizer.clear_cache()
        fusion_traj = None
        if return_traj:
            fusion_traj = {'prob_traj': torch.stack(r_prob_traj, dim=1), 'coord_traj': torch.stack(r_coord_traj, dim=1), 'design_mask': smask.clone()}
        return (H, S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd, fusion_traj)

    def forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, xloss_mask, context_ratio=0, backbone_bonds=None, complex_ids=None):
        if self.backbone_only:
            X, template = (X[:, :4], template[:, :4])
            xloss_mask = xloss_mask[:, :4]
        true_X, true_S = (X.clone(), S.clone())
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd, _ = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths)
        if getattr(self, 'paper_contract', False):
            from utils.paper_losses import paper_generator_losses
            local_indices = torch.nonzero(self.batch_constants['local_mask'], as_tuple=False).flatten()
            distance_pairs = local_indices[self.batch_constants['local_inter_edges']]
            loss_dtype = torch.float32 if pred_X.dtype in (torch.float16, torch.bfloat16) else pred_X.dtype
            with torch.autocast(device_type=pred_X.device.type, enabled=False):
                terms = paper_generator_losses(pred_X=pred_X.to(loss_dtype), true_X=true_X.to(loss_dtype), true_S=true_S, movable_mask=cmask, observed_mask=xloss_mask, batch_id=batch_id, aa_feature=self.aa_feature, backbone_bonds=backbone_bonds, sequence_rounds=[(logits.to(loss_dtype), mask) for logits, mask in r_pred_S_logits], shadow_X=r_interface_X[-1].to(loss_dtype), paratope_mask=paratope_mask, distance_rounds=[d.to(loss_dtype) for d in r_edge_dist] if self.pred_edge_dist else (), distance_pairs=distance_pairs, condition_protocol=getattr(self, 'condition_protocol', 'P2-B'), given_X=true_X.detach(), loss_mode=getattr(self, 'loss_mode', 'full'))
            self.last_supervision_counts = {scope: {name: count.detach() for name, count in values.items()} for scope, values in terms['counts'].items()}
            self.last_missing_observations = {scope: {name: count.detach() for name, count in values.items()} for scope, values in terms.get('missing_observations', {}).items()}
            self.last_coordinate_eligible = terms['coordinate_eligible'].detach()
            self._clean_batch_constants()
            with torch.no_grad():
                aa_hit = pred_S[smask] == true_S[smask]
                aar = aa_hit.float().mean() if aa_hit.numel() else pred_X.new_tensor(float('nan'))
            struct_loss = terms['coord'] + terms['bb'] + terms['sc']
            dock_loss = terms['shadow'] + terms['dist']
            return (terms['total'], (terms['seq'], aar), (struct_loss, terms['coord'], terms['bb'], terms['sc']), (dock_loss, terms['shadow'], terms['dist'], terms['distance_round_losses']), (None, None))
        snll, total = (0, 0)
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total.clamp_min(1)
        with torch.autocast(device_type=pred_X.device.type, enabled=False):
            loss_X = pred_X.float() if pred_X.dtype in (torch.float16, torch.bfloat16) else pred_X
            struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(loss_X, true_X.to(loss_X.dtype), true_S, cmask, batch_id, xloss_mask, self.aa_feature)
        gt_interface_X = true_X[paratope_mask]
        interface_atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        interface_atom_mask = interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = F.smooth_l1_loss(r_interface_X[-1][interface_atom_mask], gt_interface_X[interface_atom_mask])
        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(self.normalizer.normalize(true_X), true_S)
            ed_loss, r_ed_losses = (0, [])
            for edge_dist in r_edge_dist:
                r_ed_loss = F.smooth_l1_loss(edge_dist, gt_edge_dist)
                ed_loss = ed_loss + r_ed_loss
                r_ed_losses.append(r_ed_loss)
        else:
            r_ed_losses = [0 for _ in range(self.round)]
            ed_loss = 0
        dock_loss = interface_loss + ed_loss
        if self.struct_only:
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = (None, None)
        loss = snll + struct_loss + dock_loss + (0 if pdev_loss is None else pdev_loss)
        self._clean_batch_constants()
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]
        return (loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss))

    def sample(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None, return_hidden=False, return_traj=False, backbone_bonds=None, complex_ids=None):
        paper_contract = getattr(self, 'paper_contract', False)
        if paper_contract and (not bool(smask.any())):
            self.last_technical_failure = 'empty_design_region'
            raise ValueError('Paper sampling requires a nonempty design region')
        if self.backbone_only:
            X, template = (X[:, :4], template[:, :4])
        gen_X, gen_S = (X.clone(), S.clone())
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]
        if paper_contract and bool((torch.bincount(s_batch_id, minlength=int(batch_size)) == 0).any()):
            self.last_technical_failure = 'empty_design_region'
            raise ValueError('Paper sampling requires a nonempty design region in every complex')
        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 10000000000.0
        interface_cmask = paratope_mask[cmask]
        best_traj = None
        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd, fusion_traj = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise, return_traj=return_traj)
            if paper_contract:
                final_tensors = {'sequence logits': r_pred_S_logits[-1][0], 'main coordinates': pred_X, 'shadow coordinates': r_interface_X[-1]}
                for name, tensor in final_tensors.items():
                    if not bool(torch.isfinite(tensor).all()):
                        self.last_technical_failure = 'nonfinite_generated_output'
                        raise ValueError('Paper sampling produced non-finite ' + name)
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                metric = scatter_mean(nlls, s_batch_id)
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id)
            if paper_contract and (not bool(torch.isfinite(metric).all())):
                self.last_technical_failure = 'nonfinite_generation_score'
                raise ValueError('Paper sampling produced a non-finite generation score')
            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            best_metric[update] = metric[update]
            if return_traj and fusion_traj is not None:
                if best_traj is None:
                    best_traj = {'prob_traj': fusion_traj['prob_traj'].clone(), 'coord_traj': fusion_traj['coord_traj'].clone(), 'design_mask': fusion_traj['design_mask'].clone()}
                else:
                    best_traj['prob_traj'][update[batch_id]] = fusion_traj['prob_traj'][update[batch_id]]
                    best_traj['coord_traj'][update[batch_id]] = fusion_traj['coord_traj'][update[batch_id]]
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
            interface_X = r_interface_X[-1]
            if getattr(self, 'paper_contract', False) and getattr(self, 'condition_protocol', 'P2-B') == 'P1':
                continue
            for i in range(batch_size):
                if not update[i]:
                    continue
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                with torch.autocast(device_type=ori_cdr.device.type, enabled=False):
                    _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X
        if getattr(self, 'paper_contract', False):
            gen_X = self.aa_feature.update_globel_coordinates(gen_X, gen_S)
        self._clean_batch_constants()
        if return_hidden:
            return (gen_X, gen_S, metric, H)
        if return_traj:
            return (gen_X, gen_S, metric, best_traj)
        return (gen_X, gen_S, metric)
