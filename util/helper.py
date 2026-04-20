import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    # Python built-ins
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU & GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Make cuDNN deterministic (IMPORTANT)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make sure hash-based operations are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)

def ids_to_index_tensor(ids, id2nid, device=None):
    idx = [id2nid.get(s, -1) for s in ids]           # one-time Python cost
    t = torch.tensor(idx, dtype=torch.long, device=device)
    mask = t.ge(0)
    return t, mask

def generate_context_vec(deg_e, avg_deg_e, max_deg_e, rel_counts):
    deg_e_norm      = normalize_1d(deg_e)
    avg_deg_e_norm  = normalize_1d(avg_deg_e)
    max_deg_e_norm  = normalize_1d(max_deg_e)

    # maybe also log-scale total_pairs
    r_sub  = rel_counts[:, 0].astype(np.float32)
    r_comp = rel_counts[:, 1].astype(np.float32)
    ttl    = rel_counts[:, 2].astype(np.float32)

    p_sub  = r_sub  / (ttl + 1e-12)
    p_comp = r_comp / (ttl + 1e-12)

    # final c_e: you can choose 7 dims like this:
    c_e = np.column_stack([
        deg_e_norm,
        avg_deg_e_norm,
        max_deg_e_norm,
        p_sub,
        p_comp,
        ttl / ttl.max(),       # normalized total_pairs
        (p_sub - p_comp),      # simple sub-vs-comp balance
    ]).astype(np.float32)
    return c_e

def generate_w_relation(group2rcounts, relation):
    if relation == 'alsoBought':
        w_relation = group2rcounts[:, 1]
        # w_relation = (group2rcounts[:, 1] - group2rcounts[:, 0])/ group2rcounts[:, 2]
    elif relation == 'compared':
        w_relation = group2rcounts[:, 0]
        # w_relation = (group2rcounts[:, 0] - group2rcounts[:, 1])/ group2rcounts[:, 2]
    elif relation == 'boughtTogether':
        w_relation = group2rcounts[:, 1]
        # w_relation = (group2rcounts[:, 1] - group2rcounts[:, 0])/ group2rcounts[:, 2]
    elif relation == 'alsoViewed':
        w_relation = group2rcounts[:, 0]
        # w_relation = (group2rcounts[:, 0] + group2rcounts[:, 1])/ group2rcounts[:, 2]
        # w_relation = np.ones(group2rcounts[:, 0] .shape)
        # w_relation = None
    if w_relation is not None:
        w_relation = torch.from_numpy(w_relation).float().cuda()[:, None]
    return w_relation

def normalize_1d(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean()
    std  = x.std()
    return (x - mean) / (std + eps)

def normalize_to_range(arr, m, n):
    arr = np.array(arr, dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)
    return m + (arr - min_val) * (n - m) / (max_val - min_val)

def normalize_to_range_v2(arr, a):
    # arr assumed in [0,1]
    return a + (1 - a) * arr