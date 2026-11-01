import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
from .layers import MPGFMessageLayer

class MPGFEncoder(nn.Module):

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
            self.add_module('ctx_gcl_{0}'.format(i), MPGFMessageLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout))
            self.add_module('inter_gcl_{0}'.format(i), MPGFMessageLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout))
        self.out_layer = MPGFMessageLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual)

    def forward(self, h, x, ctx_edges, inter_mask, inter_x, inter_edges, update_mask, inter_update_mask, channel_attr, channel_weights, ctx_edge_attr=None, fusion_context=None, coordinate_callback=None, inter_coordinate_callback=None):
        h = self.linear_in(h)
        h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]
        inter_fusion_context = None if fusion_context is None else fusion_context.subset(inter_mask)
        ctx_states, ctx_coords, inter_coords = ([], [], [])
        for i in range(0, self.n_layers):
            h, x = self._modules['ctx_gcl_{0}'.format(i)](h, ctx_edges, x, channel_attr, channel_weights, edge_attr=ctx_edge_attr, fusion_context=fusion_context)
            if coordinate_callback is not None:
                x = coordinate_callback(x)
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h, inter_x = self._modules['inter_gcl_{0}'.format(i)](inter_h, inter_edges, inter_x, inter_channel_attr, inter_channel_weights, fusion_context=inter_fusion_context)
            if inter_coordinate_callback is not None:
                inter_x = inter_coordinate_callback(inter_x)
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)
        h, x = self.out_layer(h, ctx_edges, x, channel_attr, channel_weights, edge_attr=ctx_edge_attr, fusion_context=fusion_context)
        if coordinate_callback is not None:
            x = coordinate_callback(x)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return (h, x, inter_x)
