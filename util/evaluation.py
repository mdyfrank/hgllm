import torch
from collections import defaultdict
from typing import List, Dict, Iterable

@torch.no_grad()
def eval_recall_ndcg(
    g,                                   # DGL heterograph with etypes ('user','ui','item')
    user2nid: Dict[str, int],
    item2nid: Dict[str, int],
    test_users: Iterable[str],
    test_items: Iterable[str],
    user_emb: torch.Tensor,              # model.user_embedding or final h['user']
    item_emb: torch.Tensor,              # model.item_embedding or final h['item']
    Ks: Iterable[int] = (10, 20),
    batch_size: int = 1024,
    device: torch.device = None,
):
    """
    test_users/test_items are parallel lists of test interactions (user,item).
    All items interacted in TRAIN (edges in g['ui']) are removed from candidates.
    """
    assert user_emb.dim() == 2 and item_emb.dim() == 2
    num_items = item_emb.size(0)
    device = device or user_emb.device

    # 1) Build ground-truth items for each test user (skip unknowns)
    gt = defaultdict(set)  # user_nid -> set(item_nid)
    for u_str, i_str in zip(test_users, test_items):
        u = user2nid.get(u_str, None)
        i = item2nid.get(i_str, None)
        if u is not None and i is not None:
            gt[u].add(i)

    if not gt:
        raise ValueError("No valid test user-item pairs after mapping IDs.")

    # 2) Build per-user train interactions from g to filter at ranking time
    #    We fetch edges(u->i) and accumulate items per user.
    u_e, i_e = g.edges(etype=('user', 'ui', 'item'))
    u_e = u_e.tolist()
    i_e = i_e.tolist()
    train_items = defaultdict(set)  # user_nid -> set(item_nid trained on)
    for u, i in zip(u_e, i_e):
        train_items[u].add(i)

    # Candidate evaluation users
    eval_users = sorted(gt.keys())

    # Precompute ideal DCG for NDCG@K: IDCG = sum_{t=1..min(|GT|,K)} 1/log2(t+1)
    def idcg_at_k(rel_count: int, K: int) -> float:
        from math import log2
        R = min(rel_count, K)
        return sum(1.0 / log2(t + 1) for t in range(1, R + 1)) if R > 0 else 0.0

    # 3) Score in batches and compute metrics
    Ks = list(sorted(set(Ks)))
    recall_sums = {K: 0.0 for K in Ks}
    ndcg_sums   = {K: 0.0 for K in Ks}
    user_count  = 0

    item_emb = item_emb.to(device)
    user_emb = user_emb.to(device)

    for start in range(0, len(eval_users), batch_size):
        batch_users = eval_users[start:start+batch_size]
        # Gather user embedding matrix [B, D]
        u_idx = torch.tensor(batch_users, device=device, dtype=torch.long)
        U = user_emb.index_select(0, u_idx)                  # [B, D]

        # Dot product scores against all items [B, N]
        scores = U @ item_emb.t()

        # Mask train items for each user to -inf
        # This is done row-by-row because the set sizes differ
        neg_inf = torch.finfo(scores.dtype).min
        for row, u in enumerate(batch_users):
            if train_items[u]:
                idx = torch.tensor(list(train_items[u]), device=device, dtype=torch.long)
                scores[row, idx] = neg_inf

        # 4) TopK per user once for the maximum K
        maxK = max(Ks)
        topk_scores, topk_items = torch.topk(scores, k=maxK, dim=1)

        # 5) Compute per-user metrics
        for row, u in enumerate(batch_users):
            gt_items = gt[u]
            if not gt_items:
                continue
            user_count += 1

            # positions of recommended items
            rec = topk_items[row].tolist()

            for K in Ks:
                topK = rec[:K]
                hits = [1 if i in gt_items else 0 for i in topK]
                hit_count = sum(hits)

                # Recall@K
                recall_sums[K] += hit_count / float(len(gt_items))

                # NDCG@K
                # DCG: sum 1/log2(rank+1) for hits
                from math import log2
                dcg = 0.0
                for rank, is_hit in enumerate(hits, start=1):
                    if is_hit:
                        dcg += 1.0 / log2(rank + 1)
                idcg = idcg_at_k(len(gt_items), K)
                ndcg_sums[K] += (dcg / idcg) if idcg > 0 else 0.0

    # 6) Aggregate
    if user_count == 0:
        raise ValueError("No evaluable test users. Check mappings and test data.")

    out = {
        f"Recall@{K}": recall_sums[K] / user_count
        for K in Ks
    }
    out.update({
        f"NDCG@{K}": ndcg_sums[K] / user_count
        for K in Ks
    })
    return out
