#!/usr/bin/env python
# coding: utf-8



from util.load_data import load_test_file, build_user_item_graph, construct_negative_graph, construct_user_item_bigraph
from util.helper import set_seed, ids_to_index_tensor
from util.evaluation import eval_recall_ndcg
import argparse
import torch
from model import HGCN




def parse_args():
    parser = argparse.ArgumentParser(description="Recommendation Model Configuration")

    # Define arguments
    parser.add_argument("--dataset", type=str, default="instacart", help="dataset name")
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--emb_dim", type=int, default=64, help="embedding size")
    parser.add_argument("--norm", type=float, default=-1, help="learning rate")
    parser.add_argument("--layer_num", type=int, default=2, help="learning rate")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--neg_samples", type=int, default=1, help="learning rate")
    parser.add_argument("--regularization", type=float, default=1e-4, help="learning rate")
    
    args, _ = parser.parse_known_args()
    
    return vars(args)




dataset = 'us_electronics'
train_file = "us_electronics.train.inter"
train_path = f"./datasets/{dataset}/{train_file}"
test_file = "us_electronics.test.inter"
test_path = f"./datasets/{dataset}/{test_file}"
test_users, test_items = load_test_file(test_path)
set_seed(2025)
g, user2nid, item2nid = build_user_item_graph(train_path)
print(g)
print("#users:", g.num_nodes('user'), "#items:", g.num_nodes('item'), "#buys edges:", g.num_edges('ui'))



device = 'cuda:{}'.format(0)
args = parse_args()
g = g.to(device)
pos_g = construct_user_item_bigraph(g)
model = HGCN(g, args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])




user_idx_t, user_mask = ids_to_index_tensor(test_users, user2nid, device=model.user_embedding.device)
embs = model.user_embedding.index_select(0, user_idx_t[user_mask])




for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)




for epoch in range(args['epochs']):
    neg_g = construct_negative_graph(g, args['neg_samples'], device=device)
    emb_h = model(g)
    bpr_loss, mf_loss, emb_loss = model.create_bpr_loss(pos_g, neg_g, emb_h)
    print(bpr_loss)
    optimizer.zero_grad()
    bpr_loss.backward()
    optimizer.step()
    # print(model.user_embedding)



model.eval()
with torch.no_grad():
    user_emb = emb_h['user']; item_emb = emb_h['item']
metrics = eval_recall_ndcg(
    g=g,
    user2nid=user2nid,
    item2nid=item2nid,
    test_users=test_users,   # parallel to test_items
    test_items=test_items,
    user_emb=user_emb,
    item_emb=item_emb,
    Ks=(10, 20),
    batch_size=2048,
    device=user_emb.device
)

print(metrics)


