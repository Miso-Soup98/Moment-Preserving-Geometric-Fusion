import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax
from utils.singleton import singleton

class MPGFGraphNetwork(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf, radial_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, residual=True, dropout=0.1, dense=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module('gcl_{0}'.format(i), MPGFMessageLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout))
        self.out_layer = MPGFMessageLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual)

    def forward(self, h, x, edges, channel_attr, channel_weights, ctx_edge_attr=None, fusion_context=None, coordinate_callback=None):
        h = self.linear_in(h)
        h = self.dropout(h)
        ctx_states, ctx_coords = ([], [])
        for i in range(0, self.n_layers):
            h, x = self._modules['gcl_{0}'.format(i)](h, edges, x, channel_attr, channel_weights, edge_attr=ctx_edge_attr, fusion_context=fusion_context)
            if coordinate_callback is not None:
                x = coordinate_callback(x)
            ctx_states.append(h)
            ctx_coords.append(x)
        h, x = self.out_layer(h, edges, x, channel_attr, channel_weights, edge_attr=ctx_edge_attr, fusion_context=fusion_context)
        if coordinate_callback is not None:
            x = coordinate_callback(x)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return (h, x)

@singleton
class MPGFRollerPooling(nn.Module):

    def __init__(self, n_channel) -> None:
        super().__init__()
        self.n_channel = n_channel
        with torch.no_grad():
            pool_matrix = []
            ones = torch.ones((n_channel, n_channel), dtype=torch.float)
            for i in range(n_channel):
                window_size = n_channel - i
                mat = torch.triu(ones) - torch.triu(ones, diagonal=window_size)
                pool_matrix.append(mat / window_size)
            self.pool_matrix = torch.stack(pool_matrix)

    def forward(self, hidden, target_size):
        pool_mat = self.pool_matrix.to(hidden.device).type(hidden.dtype)
        pool_mat = pool_mat[target_size - 1]
        hidden = hidden.unsqueeze(-1)
        return torch.bmm(pool_mat, hidden)

class MPGFMessageLayer(nn.Module):
    _mpgf_owned_message_layer = True

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, channel_nf, radial_nf, edges_in_d=0, node_attr_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, dropout=0.1):
        super(MPGFMessageLayer, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-08
        self.dropout = nn.Dropout(dropout)
        input_edge = input_nf * 2
        self.edge_mlp = nn.Sequential(nn.Linear(input_edge + radial_nf + edges_in_d, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.radial_linear = nn.Linear(channel_nf ** 2, radial_nf)
        self.node_mlp = nn.Sequential(nn.Linear(hidden_nf + input_nf + node_attr_d, hidden_nf), act_fn, nn.Linear(hidden_nf, output_nf))
        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return (out, agg)

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, channel_weights):
        row, col = edge_index
        n_channel = channel_weights.shape[-1]
        edge_feat = self.coord_mlp(edge_feat)
        channel_sum = (channel_weights != 0).long().sum(-1)
        pooled_edge_feat = MPGFRollerPooling(n_channel)(edge_feat, channel_sum[row])
        trans = coord_diff * pooled_edge_feat
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, channel_attr, channel_weights, edge_attr=None, node_attr=None, fusion_context=None):
        row, col = edge_index
        if getattr(self, 'paper_contract', False) and bool(((channel_weights != 0).sum(-1) == 0).any()):
            self.last_technical_failure = 'all_zero_coordinate_channels'
            raise ValueError('Paper coordinate pathway requires at least one nonzero channel per participating node')
        if fusion_context is None:
            radial, coord_diff = coord2radial(edge_index, coord, channel_attr, channel_weights, self.radial_linear)
        else:
            from mpgf.adapters import dispatch_geometry
            plan = fusion_context.edge_plan(edge_index)
            radial, coord_diff = coord2radial(edge_index, coord, channel_attr, channel_weights, self.radial_linear, radial_edge_index=plan.auxiliary_edges)
            radial = dispatch_geometry(self.fusion, fusion_context, edge_index, coord, channel_attr, channel_weights, radial, plan=plan)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, channel_weights)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return (h, coord)

def unsorted_segment_sum(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments,) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
CONSTANT = 1
NUM_SEG = 1

def coord2radial(edge_index, coord, attr, channel_weights, linear_map, radial_edge_index=None):
    row, col = edge_index
    radial_row, radial_col = edge_index if radial_edge_index is None else radial_edge_index
    radials = []
    seg_size = (len(radial_row) + NUM_SEG - 1) // NUM_SEG
    for i in range(NUM_SEG):
        start = i * seg_size
        end = min(start + seg_size, len(radial_row))
        if end <= start:
            break
        seg_row, seg_col = (radial_row[start:end], radial_col[start:end])
        coord_msg = torch.norm(coord[seg_row].unsqueeze(2) - coord[seg_col].unsqueeze(1), dim=-1, keepdim=False)
        coord_msg = coord_msg * torch.bmm(channel_weights[seg_row].unsqueeze(2), channel_weights[seg_col].unsqueeze(1))
        radial = torch.bmm(attr[seg_row].transpose(-1, -2), coord_msg)
        radial = torch.bmm(radial, attr[seg_col])
        radial = radial.reshape(radial.shape[0], -1)
        radial_norm = torch.norm(radial, dim=-1, keepdim=True) + CONSTANT
        radial = linear_map(radial) / radial_norm
        radials.append(radial)
    radials = torch.cat(radials, dim=0) if radials else linear_map(coord.new_empty((0, attr.shape[-1] ** 2)))
    if radial_edge_index is not None and len(radial_row) == 0:
        dependency = (coord[:1, :1, :1].sum() + attr[:1, :1, :1].sum() + channel_weights[:1, :1].sum()) * 0
        radials = radials + dependency
    channel_mask = (channel_weights != 0).long()
    channel_sum = channel_mask.sum(-1)
    pooled_col_coord = (coord[col] * channel_mask[col].unsqueeze(-1)).sum(1)
    pooled_col_coord = pooled_col_coord / channel_sum[col].unsqueeze(-1)
    coord_diff = coord[row] - pooled_col_coord.unsqueeze(1)
    return (radials, coord_diff)
