"""Citation-influence edge classifier.

encoder: input projection -> L x (GraphSAGE + LayerNorm + residual), output = [layer0 ; layerL]
         (message passing skipped in the MLP baseline)
decoder: MLP([h_src, h_dst, h_src*h_dst, |h_src-h_dst|, edge_features]) -> logit(influential)
"""
from __future__ import annotations

import torch
from torch import nn


class InfluenceModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 256, layers: int = 2,
                 dropout: float = 0.3, use_graph: bool = True, use_edge: bool = True):
        super().__init__()
        self.cfg = dict(node_dim=node_dim, edge_dim=edge_dim, hidden=hidden, layers=layers,
                        dropout=dropout, use_graph=use_graph, use_edge=use_edge)
        self.inp = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.use_graph, self.use_edge = use_graph, use_edge
        if use_graph:
            from torch_geometric.nn import SAGEConv
            self.convs = nn.ModuleList([SAGEConv(hidden, hidden, aggr="mean") for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        # jumping-knowledge: the decoder sees both the raw (layer-0) and the message-passed representation,
        # so the GNN can always fall back to what the MLP baseline sees
        rep = 2 * hidden if use_graph else hidden
        dec_in = 4 * rep + (edge_dim if use_edge else 0)
        self.dec = nn.Sequential(nn.Linear(dec_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h0 = h = self.inp(x)
        if not self.use_graph:
            return h0
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.drop(torch.nn.functional.gelu(norm(conv(h, edge_index))))
        return torch.cat([h0, h], -1)

    def decode(self, h, src, dst, eattr):
        hs, hd = h[src], h[dst]
        z = [hs, hd, hs * hd, (hs - hd).abs()]
        if self.use_edge:
            z.append(eattr)
        return self.dec(torch.cat(z, -1)).squeeze(-1)

    def forward(self, x, edge_index, src, dst, eattr):
        return self.decode(self.encode(x, edge_index), src, dst, eattr)
