from __future__ import print_function
import time
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
# --- Make sure this import points to your corrected Dataset_Loader ---
from local_code.stage_5_code.source_code_citeseer.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.source_code_citeseer.models import GCN
from local_code.stage_5_code.source_code_citeseer.train import train, plot_learning_curves, test

seed = 16

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=seed, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
loss = []

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

curr = Dataset_Loader()
curr.dataset_name = 'citeseer'
curr.dataset_source_folder_path = "../../data/stage_5_data/citeseer/"
adj, features, labels, idx_train, idx_val, idx_test = curr.load()

# Ensure you are using the 2-layer GCN model
model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

t_total = time.time()
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print("Starting training with corrected data preprocessing...")
for epoch in range(args.epochs):
    loss_val = train(epoch, adj, features, labels, idx_train, idx_val, idx_test, model, optimizer, args, loss)
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= args.patience:
        print(f"Early stopping at epoch {epoch + 1}!")
        break

if best_model_state:
    print("Loading best model for testing...")
    model.load_state_dict(best_model_state)

plot_learning_curves(len(loss), loss)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

test(adj, features, labels, idx_train, idx_val, idx_test, model)