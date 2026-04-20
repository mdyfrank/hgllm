import dgl
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

def restrict_subgraph_to_user_items(subg: dgl.DGLHeteroGraph,
                                    g_user_item: dgl.DGLHeteroGraph,
                                    relation: str,
                                    device=None) -> dgl.DGLHeteroGraph:
    """
    Keep only items in `subg` that are present in the user–item graph `g_user_item`.
    Then drop relation nodes with zero degree. Works for bipartite (item <-> relation) subgraphs.

    Requirement: both graphs share the same global item ID space (same item2nid).
    """
    device = device or next(iter(subg.ntypes)).__class__ and torch.device("cpu")
    subg = subg.to(device)
    g_user_item = g_user_item.to(device)

    # 1) Build mask for items to keep in subgraph
    # subgraph item global IDs:
    item_NID = subg.nodes['item'].data[dgl.NID].to(device)  # [n_item_sub]

    # CASE A (typical): g has items 0..N-1 with same mapping -> quick mask
    num_items_g = g_user_item.num_nodes('item')
    mask_item = item_NID < num_items_g

    # If your item IDs are not guaranteed contiguous or identical, use a set check instead:
    # allowed = torch.arange(num_items_g, device=device)  # or a tensor of allowed global IDs
    # mask_item = torch.isin(item_NID, allowed)

    # 2) Keep all relation nodes for now (we’ll prune isolated ones after)
    mask_rel = torch.ones(subg.num_nodes(relation), dtype=torch.bool, device=device)

    subg1 = dgl.node_subgraph(subg, {'item': mask_item, relation: mask_rel})

    # 3) Drop relation nodes that have no remaining edges to items
    # Out-degree of relation nodes along (relation -> item) edges
    deg_rel = subg1.out_degrees(etype=(relation, f'{relation}-item', 'item'))
    mask_rel2 = deg_rel > 0
    if mask_rel2.sum().item() < mask_rel2.numel():
        subg2 = dgl.node_subgraph(subg1, {'item': torch.ones(subg1.num_nodes('item'), dtype=torch.bool, device=device),
                                          relation: mask_rel2})
    else:
        subg2 = subg1

    return subg2

def load_test_file(path):
    df = pd.read_csv(path, sep=" ")
    user_ids = df['user_id:token'].tolist()
    item_ids = df['item_id:token'].tolist()
    return user_ids, item_ids

def build_user_item_graph(path):
    """Read a space-separated file 'user item' → DGL heterograph with
    etypes: ('user', 'buys', 'item') and ('item', 'bought-by', 'user').
    Returns (g, user2nid, item2nid).
    """
    user2nid, item2nid = {}, {}
    u_nxt, i_nxt = 0, 0

    # collect raw edges (string IDs)
    raw_edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # format: user_id:token item_id:token  (tokens may already be alphanum)
            try:
                u_str, i_str = line.split()
            except ValueError:
                continue  # skip malformed lines

            # optional: lines might include "user_id:token" literal key; handle both
            if ":" in u_str:
                # e.g., "AHCG...": no colon; or "user_id:token"? If the literal
                # keys appear (e.g., "user_id:token"), they will be followed by real tokens.
                # For your example, there is no literal prefix, so nothing to do.
                continue
            raw_edges.append((u_str, i_str))

    # map strings to contiguous node ids
    u_ids, i_ids = [], []
    for u_str, i_str in raw_edges:
        if u_str not in user2nid:
            user2nid[u_str] = u_nxt; u_nxt += 1
        if i_str not in item2nid:
            item2nid[i_str] = i_nxt; i_nxt += 1
        u_ids.append(user2nid[u_str])
        i_ids.append(item2nid[i_str])

    # deduplicate edges (common in interaction logs)
    edge_set = set(zip(u_ids, i_ids))
    if len(edge_set) != len(u_ids):
        u_ids, i_ids = zip(*sorted(edge_set))
    else:
        # ensure tensors even if already unique
        u_ids, i_ids = torch.tensor(u_ids), torch.tensor(i_ids)

    if not isinstance(u_ids, torch.Tensor):
        u_ids, i_ids = torch.tensor(u_ids), torch.tensor(i_ids)

    num_users = len(user2nid)
    num_items = len(item2nid)

    # build heterograph
    data_dict = {
        ('user', 'ui', 'item'): (u_ids, i_ids),
        ('item', 'iu', 'user'): (i_ids, u_ids),
    }
    g = dgl.heterograph(data_dict, num_nodes_dict={'user': num_users, 'item': num_items})

    # (optional) add degree features or counts
    # g.nodes['user'].data['deg'] = dgl.ops.copy_u_sum(g['buys'], torch.ones(g.num_edges('buys')))
    # g.nodes['item'].data['deg'] = dgl.ops.copy_v_sum(g['buys'], torch.ones(g.num_edges('buys')))

    return g, user2nid, item2nid

def construct_user_item_bigraph(graph):
    return graph.node_type_subgraph(['user', 'item'])

def construct_negative_graph(graph, k, device):
    user_item_src, _ = graph.edges(etype='ui')
    neg_src = user_item_src.repeat_interleave(k)
    n_neg_src = len(user_item_src)
    neg_dst = torch.randint(0, graph.num_nodes(ntype='item'), (n_neg_src * k,)).to(device)
    data_dict = {
        ('user', 'ui', 'item'): (neg_src, neg_dst),
        ('item', 'iu', 'user'): (neg_dst, neg_src),
        # ('neg_user', 'ui', 'side'): (user_side_src, user_side_dst),
        # ('side', 'iu', 'neg_user'): (user_side_dst, user_side_src)
    }
    num_dict = {
        'user': graph.num_nodes(ntype='user'), 'item': graph.num_nodes(ntype='item'),
        # 'side': graph.num_nodes(ntype='side')
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict)


@torch.no_grad()
def construct_true_negative_graph(graph, k: int, device):
    """
    For each positive ('user','ui','item') edge (u, i_pos), sample k negatives (u, i_neg)
    such that (u, i_neg) was NOT observed in train.
    Returns a heterograph with the negative edges.
    """
    # Positive edges
    u_pos, i_pos = graph.edges(etype=('user', 'ui', 'item'))
    u_pos = u_pos.to(device)
    i_pos = i_pos.to(device)

    num_items = graph.num_nodes('item')

    # Build the set of positive keys to forbid: key = u * num_items + i
    pos_keys = (u_pos * num_items + i_pos)
    # use sorted unique tensor for O(log n) membership checks via searchsorted
    pos_keys = pos_keys.unique().sort().values

    # Repeat users k times: we will sample k negatives per positive edge
    neg_src = u_pos.repeat_interleave(k)                 # [E*k]
    n = neg_src.numel()

    # Initial random negatives
    neg_dst = torch.randint(0, num_items, (n,), device=device)  # [E*k]

    # Helper: check membership in pos_keys using searchsorted (faster than torch.isin for large tensors)
    def in_pos(keys: torch.Tensor, sorted_pos: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(sorted_pos, keys)
        in_range = idx < sorted_pos.numel()
        return in_range & (sorted_pos[idx.clamp_max(sorted_pos.numel()-1)] == keys)

    # Repair conflicts by resampling only the conflicting positions
    neg_keys = neg_src * num_items + neg_dst
    conflict = in_pos(neg_keys, pos_keys)
    # In extremely dense cases this loop may iterate a few times; in sparse rec logs it exits quickly.
    while conflict.any():
        cnt = int(conflict.sum().item())
        neg_dst[conflict] = torch.randint(0, num_items, (cnt,), device=device)
        neg_keys = neg_src * num_items + neg_dst
        conflict = in_pos(neg_keys, pos_keys)

    data_dict = {
        ('user', 'ui', 'item'): (neg_src, neg_dst),
        ('item', 'iu', 'user'): (neg_dst, neg_src),
    }
    num_dict = {
        'user': graph.num_nodes('user'),
        'item': graph.num_nodes('item'),
    }
    return dgl.heterograph(data_dict, num_nodes_dict=num_dict).to(device)

def build_item_group_graph(
    path: str,
    item2nid: Dict[str, int],
    min_group_size: int = 2,
    device: str = "cpu",
) -> Tuple[dgl.DGLHeteroGraph, Dict[str, int], Dict[int, List[int]], np.ndarray]:
    """
    Read 'alsoBought.txt' where each line is a group of item IDs separated by spaces.
    Build a bipartite heterograph with node types 'item' and 'group':
        ('item','in_group','group') and ('group','has_item','item').
    Args
    ----
    path : file path to alsoBought.txt
    item2nid : optional existing mapping item_id -> item node index; if None, will build
    min_group_size : drop groups smaller than this (e.g., 1)
    device : 'cpu' or 'cuda:0', etc.

    Returns
    -------
    g : DGLHeteroGraph
    item2nid : mapping used (possibly extended)
    group2items : dict gid -> list[item_nid] (for reference/debug)
    """
    count = 1 if 'counts' in path else 0
    relation = path.split('/')[-1].split('.')[0]
    print(f'creating item-item relation graph on {relation}...')
    # 1) parse file, collect groups (as lists of string item IDs)
    groups: List[List[str]] = []
    sub_list, com_list, ttl_list = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            items = line.split()
            # remove duplicates within a line but keep order
            seen = set()
            if count:
                valid_items = items[:-3]
                sub, com, ttl = items[-3:]
                items = valid_items
            uniq_items = [x for x in items if (not (x in seen or seen.add(x)) and (x in item2nid))]
            if len(uniq_items) >= min_group_size:
                groups.append(uniq_items)
                if count:
                    sub_list.append(int(sub))
                    com_list.append(int(com))
                    ttl_list.append(int(ttl))


    num_items = len(item2nid)
    num_groups = len(groups)

    # 3) build edge lists
    item_src, group_dst = [], []  # item -> group
    group_src, item_dst = [], []  # group -> item
    group2items: Dict[int, List[int]] = {}
    group2relation_counts = np.zeros((num_groups, 3))

    for gid, items in enumerate(groups):
        nid_list = [item2nid[it] for it in items]
        group2items[gid] = nid_list
        if count:
            group2relation_counts[gid] = [sub_list[gid], com_list[gid], ttl_list[gid]]
        for nid in nid_list:
            item_src.append(nid); group_dst.append(gid)
            group_src.append(gid); item_dst.append(nid)

    # 4) deduplicate edges (optional; harmless if no dup)
    edge_set_ig = set(zip(item_src, group_dst))
    edge_set_gi = set(zip(group_src, item_dst))
    if len(edge_set_ig) != len(item_src):
        item_src, group_dst = zip(*sorted(edge_set_ig))
    if len(edge_set_gi) != len(group_src):
        group_src, item_dst = zip(*sorted(edge_set_gi))

    # 5) tensors
    item_src = torch.as_tensor(item_src, dtype=torch.long)
    group_dst = torch.as_tensor(group_dst, dtype=torch.long)
    group_src = torch.as_tensor(group_src, dtype=torch.long)
    item_dst = torch.as_tensor(item_dst, dtype=torch.long)

    # 6) build heterograph
    data_dict = {
        ('item',  f'item-{relation}', f'{relation}'): (item_src, group_dst),
        (f'{relation}', f'{relation}-item', 'item' ): (group_src, item_dst),
    }
    num_nodes_dict = {'item': num_items, f'{relation}': num_groups}
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict).to(device)
    return g, item2nid, group2items,group2relation_counts


def compute_group_level_stats(g, group2items, edge_type):
    num_groups = len(group2items)
    deg_e = np.zeros(num_groups, dtype=np.float32)
    avg_item_degree = np.zeros(num_groups, dtype=np.float32)
    max_item_degree = np.zeros(num_groups, dtype=np.float32)

    for gid, item_list in group2items.items():
        arr = np.asarray(item_list, dtype=np.int64)
        deg_e[gid] = len(arr)

        if len(arr) > 0:
            deg_vals = g.in_degrees(arr, edge_type)
            avg_item_degree[gid] = deg_vals.float().mean()
            max_item_degree[gid] = deg_vals.max()
        else:
            avg_item_degree[gid] = 0.0
            max_item_degree[gid] = 0.0

    return deg_e, avg_item_degree, max_item_degree

def load_asin2title(metadata_path: str) -> Dict[str, str]:
    """Load asin -> title mapping from metadata_us_Electronics.json (JSONL format)."""
    asin2title = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            asin = obj.get("asin")
            title = obj.get("title", "").strip()
            if asin:
                # fallback to asin if title is missing
                asin2title[asin] = title if title else asin
    return asin2title


def build_item_embedding_table(
    item2nid: Dict[str, int],
    metadata_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build an item embedding table using SentenceTransformer.
    
    Returned array `emb_table` has shape (num_items, dim), where
    `emb_table[nid]` corresponds to the item whose id is `nid` in item2nid.
    """
    # 1. load text (titles) for each asin
    asin2title = load_asin2title(metadata_path)

    # 2. prepare texts in nid order
    num_items = max(item2nid.values()) + 1
    texts: List[str] = [""] * num_items
    for asin, nid in item2nid.items():
        text = asin2title.get(asin)
        if not text:
            # if no metadata entry, fallback to asin string
            text = asin
        texts[nid] = text

    # 3. encode with SentenceTransformer
    model = SentenceTransformer(model_name)
    emb_table = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    return emb_table
