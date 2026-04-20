
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl
import numpy as np
import dgl.function as fn

def pairwise_uniformity_loss(Z: torch.Tensor, t: float = 2.0, max_samples: int = 4096) -> torch.Tensor:
    """
    Z: [N, D] l2-normalized.
    Computes log mean exp(-t * ||zi - zj||^2) over all pairs (or a random subset if N is large).
    O(N^2) if N <= max_samples, otherwise we subsample for stability.
    """
    N = Z.size(0)
    if N > max_samples:
        idx = torch.randperm(N, device=Z.device)[:max_samples]
        Z = Z.index_select(0, idx)
        N = Z.size(0)

    # squared Euclidean = 2 - 2 * cos(z_i, z_j) when z are normalized
    S = Z @ Z.t()                       # [N, N]
    d2 = 2 - 2 * S                      # [N, N], diag=0
    # exclude diagonal to avoid trivial zeros
    mask = ~torch.eye(N, dtype=torch.bool, device=Z.device)
    val = torch.exp(-t * d2[mask])
    return torch.log(val.mean() + 1e-12)

def alignment_loss(u: torch.Tensor, v: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    u, v: [B, D] positive user/item embeddings (NOT yet normalized).
    """
    u = F.normalize(u, dim=1)
    v = F.normalize(v, dim=1)
    return ((u - v).norm(p=2, dim=1).pow(alpha)).mean()

class HGCN(nn.Module):
    def __init__(self, graph, relations, subgraphs, gate_num, context_dim, args):
        super().__init__()
        self.hid_dim = args['emb_dim']
        self.norm = args['norm']
        self.layers = args['layer_num']
        self.LGCNlayers = args['LGCNlayer_num']
        self.decay = args['regularization']
        self.c_dim = args['c_dim']
        self.neg_samples = args['neg_samples']

        self.user_embedding = nn.Parameter(torch.randn(graph.nodes('user').shape[0], self.hid_dim))
        self.item_embedding = nn.Parameter(torch.randn(graph.nodes('item').shape[0], self.hid_dim))
        self.gamma = torch.tensor(1.0, requires_grad=True) 

        # keep these as buffers if you want them around but not trainable
        self.register_buffer('user_hyperedge',
                             torch.empty(graph.nodes('user').shape[0], self.hid_dim))
        self.register_buffer('item_hyperedge',
                             torch.empty(graph.nodes('item').shape[0], self.hid_dim))
        
        for idx, subgraph in enumerate(subgraphs):
            relation = relations[idx]
            self.register_buffer(f'{relation}_hyperedge', 
                                 torch.empty(subgraph.nodes(f'{relation}').shape[0], self.hid_dim))
        self.gate_num = gate_num

        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        self.context_dim = context_dim
        self.build_model()
        self.pred = ScorePredictor()

    def build_model(self):
        self.HGCNlayer = HGCNLayer()
        self.LGCNlayer = LightGCNLayer()
        if self.c_dim > 0:
            self.gate_list  = nn.ModuleList()
            for _ in range(self.gate_num):
                self.gate_list.append(HyperedgeGating(in_dim = self.context_dim, hidden_dim=self.c_dim))

    
    def forward(self, graph, h = None):
        # fresh dict each call; nothing stored back on self.*
        if h is None:
            h = {
                'user': self.user_embedding,
                'item': self.item_embedding,
            }
        for _ in range(self.layers):
            h_user, _ = self.HGCNlayer(graph, h, ('user', 'ui', 'item'), ('item', 'iu', 'user'), self.norm)
            h_item, _ = self.HGCNlayer(graph, h, ('item', 'iu', 'user'), ('user', 'ui', 'item'), self.norm)
            h['user'], h['item'] = h_user, h_item
        return h
    
    def lgcn_forward(self, graph, h):
        h_user_list, h_item_list = [], []
        for _ in range(self.LGCNlayers):
            h_item = self.LGCNlayer(graph, h, ('user', 'ui', 'item'), self.norm)
            h_user = self.LGCNlayer(graph, h, ('item', 'iu', 'user'), self.norm)
            h['user'], h['item'] = h_user, h_item
            h_user_list.append(h_user)
            h_item_list.append(h_item)
        h_item_avg = torch.stack(h_item_list, dim=0).mean(dim=0)
        h_user_avg = torch.stack(h_user_list, dim=0).mean(dim=0)
        h['user'], h['item'] = h_user_avg, h_item_avg
        return h

    def subgraph_forward(self, subgraph, relation, item_embedding = None, w_relation = None):
        h = {
            'item': item_embedding,
             relation: getattr(self, f'{relation}_hyperedge')
        }
        for _ in range(self.layers):
            h_item, _ = self.HGCNlayer(subgraph, h, ('item', f'item-{relation}', relation), (relation, f'{relation}-item', 'item'), self.norm, w_edge = w_relation)
            h['item'] = h_item
        return h
    
    def create_bpr_loss(self, pos_g, neg_g, h):
        sub_fig_feature = {'user': h['user'], 'item': h['item']}
        pos_score = self.pred(pos_g, sub_fig_feature)
        neg_score = self.pred(neg_g, sub_fig_feature)
        pos_score = pos_score[('user', 'ui', 'item')].repeat_interleave(self.neg_samples, dim=0)
        neg_score = neg_score[('user', 'ui', 'item')]
        mf_loss = nn.Softplus()(neg_score - pos_score)
        mf_loss = mf_loss.mean()
        regularizer = (1 / 2) * (self.user_embedding.norm(2).pow(2) +
                                     self.item_embedding.norm(2).pow(2))
        emb_loss = self.decay * regularizer

        bpr_loss = mf_loss + emb_loss
        return bpr_loss, mf_loss, emb_loss
    

    # --- Your method with alignment/uniformity added ---
    def create_bpr_align_loss(self, pos_g, neg_g, h, align_w: float = 1e-2, unif_w: float = 0.0,
                            unif_t: float = 2.0):
        """
        Returns:
        total_loss, dict of components
        Components:
        bpr: Softplus(neg - pos) + L2-emb
        align: alignment on positives
        unif: uniformity on a batch of (u_pos, i_pos)
        """
        # ---------- BPR part (your original) ----------
        sub_feat = {'user': h['user'], 'item': h['item']}
        pos_score = self.pred(pos_g, sub_feat)
        neg_score = self.pred(neg_g, sub_feat)
        pos_score = pos_score[('user', 'ui', 'item')].repeat_interleave(self.neg_samples, dim=0)
        neg_score = neg_score[('user', 'ui', 'item')]
        mf_loss = nn.Softplus()(neg_score - pos_score).mean()

        regularizer = 0.5 * (self.user_embedding.norm(2).pow(2) + self.item_embedding.norm(2).pow(2))
        emb_loss = self.decay * regularizer
        bpr_loss = mf_loss + emb_loss

        # ---------- Alignment on positive edges ----------
        u_idx, i_idx = pos_g.edges(etype=('user', 'ui', 'item'))
        u_pos = h['user'][u_idx]    # [E_pos, D]
        i_pos = h['item'][i_idx]    # [E_pos, D]

        align = alignment_loss(u_pos, i_pos, alpha=2.0) if align_w > 0 else torch.tensor(0.0, device=u_pos.device)

        # ---------- (Optional) Uniformity to avoid collapse ----------
        if unif_w > 0:
            # use a batch combined from the positive endpoints
            Z = torch.cat([F.normalize(u_pos, dim=1), F.normalize(i_pos, dim=1)], dim=0)
            unif = pairwise_uniformity_loss(Z, t=unif_t)
        else:
            unif = torch.tensor(0.0, device=u_pos.device)

        total = bpr_loss + align_w * align + unif_w * unif

        stats = {
            'total': float(total.detach().cpu()),
            'bpr': float(bpr_loss.detach().cpu()),
            'mf': float(mf_loss.detach().cpu()),
            'emb_reg': float(emb_loss.detach().cpu()),
            'align': float(align.detach().cpu()),
            'unif': float(unif.detach().cpu()),
        }
        return total, stats

class HyperedgeGating(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, c_e, alpha = 0.5):
        """
        c_e: (num_groups, in_dim)
        returns s_e: (num_groups,) positive scaling around 1.0
        """
        # (num_groups, 1)
        eps = self.mlp(c_e)
        eps = (eps - eps.mean()) / (eps.std() + 1e-12)
        # bound the correction so it does not explode
        # e.g. s_e in [0.5, 1.5]
        s_e = 1.0 + alpha * eps  # still (num_groups, 1)
        return s_e.squeeze(-1)
    
class HGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype_forward, etype_back, norm_2=-1, w_edge=None):
        with graph.local_scope():
            src, _, dst = etype_forward
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_u('h', 'm')
            aggregate_fn_back = fn.copy_u('h_b', 'm_b')

            graph.nodes[src].data['h'] = feat_src
            # graph.nodes[src].data['h'] = self.dropout(feat_src)
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype_forward)

            rst = graph.nodes[dst].data['h']
            in_degrees = graph.in_degrees(etype=etype_forward).float().clamp(min=1)
            norm_dst = torch.pow(in_degrees, -1)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)
            if w_edge is not None:
                rst  = rst * w_edge
            rst = rst * norm
            graph.nodes[dst].data['h_b'] = rst
            graph.update_all(aggregate_fn_back, fn.sum(msg='m_b', out='h_b'), etype=etype_back)
            bsrc = graph.nodes[src].data['h_b']

            in_degrees_b = graph.in_degrees(etype=etype_back).float().clamp(min=1)
            norm_src = torch.pow(in_degrees_b, norm_2)
            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            bsrc = bsrc * norm_src
            return bsrc, rst
    
class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype, norm_2 = -1):
        with graph.local_scope():
            src, _, dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            aggregate_fn = fn.copy_u('h', 'm')

            out_degrees = graph.out_degrees(etype=etype).float().clamp(min=1)
            norm_src = torch.pow(out_degrees, norm_2)

            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            feat_src = feat_src * norm_src

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype)

            rst = graph.nodes[dst].data['h']
            in_degrees = graph.in_degrees(etype=etype).float().clamp(min=1)
            norm_dst = torch.pow(in_degrees, norm_2)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)
            rst = rst * norm

            return rst


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']