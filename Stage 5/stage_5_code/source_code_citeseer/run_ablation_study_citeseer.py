import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

# 假设这些文件与本脚本在同一目录下
from local_code.stage_5_code.source_code_citeseer.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.source_code_citeseer.train import train, test
from local_code.stage_5_code.source_code_citeseer.layers import GraphConvolution


# --- Step 1: Define multiple GCN models locally for easy selection ---

class GCN2Layer(nn.Module):
    """Standard 2-Layer GCN Model"""

    def __init__(self, nfeat, nhid, nclass, dropout, activation=F.relu):
        super(GCN2Layer, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x, adj):
        x = self.activation(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCN3Layer(nn.Module):
    """A Deeper 3-Layer GCN Model"""

    def __init__(self, nfeat, nhid, nclass, dropout, activation=F.relu):
        super(GCN3Layer, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)  # Extra hidden layer
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.activation = activation

    def forward(self, x, adj):
        x = self.activation(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.activation(self.gc2(x, adj))  # Activation after second layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)


# --- Step 2: Update the experiment runner to be more flexible ---
def run_experiment(params):
    seed = params.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    curr = Dataset_Loader()
    curr.dataset_name = 'citeseer'
    curr.dataset_source_folder_path = "../../../data/stage_5_data/citeseer/"
    adj, features, labels, idx_train, idx_val, idx_test = curr.load()

    # Select model class based on parameters
    model_class = params.get('model_class', GCN2Layer)
    model = model_class(nfeat=features.shape[1],
                        nhid=params.get('hidden', 128),
                        nclass=labels.max().item() + 1,
                        dropout=params.get('dropout', 0.5),
                        activation=params.get('activation', F.relu))

    # Select optimizer based on parameters
    optimizer_name = params.get('optimizer_name', 'adam').lower()
    lr = params.get('lr', 0.001)
    weight_decay = params.get('weight_decay', 5e-4)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' not supported.")

    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(params.get('epochs', 200)):
        loss = []
        val_loss = train(epoch, adj, features, labels, idx_train, idx_val, idx_test, model, optimizer,
                         argparse.Namespace(fastmode=False), loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} for experiment '{params['name']}'")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    output = model(features, adj)
    preds = output[idx_test].max(1)[1].cpu().numpy()
    labels_test = labels[idx_test].cpu().numpy()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(labels_test, preds)
    precision = precision_score(labels_test, preds, average='macro', zero_division=0)
    recall = recall_score(labels_test, preds, average='macro', zero_division=0)
    f1 = f1_score(labels_test, preds, average='macro', zero_division=0)

    return {
        "Changes Made": params['name'],
        "Accuracy": acc,
        "Other Metrics": f"precision= {precision:.4f}, recall= {recall:.4f}, f1 score= {f1:.4f}"
    }


# --- Step 3: Add new experiments to the list ---
experiments = [
    # --- Original Experiments from your table ---
    {
        'name': 'Learning rate = 0.01, dropout = 0.5',
        'lr': 0.01, 'dropout': 0.5
    },
    {
        'name': 'Hidden Units = 200, dropout = 0.5',
        'hidden': 200, 'dropout': 0.5
    },
    {
        'name': 'Input Layer Activation Function = Tanh',
        'activation': torch.tanh
    },
    # --- Baseline ---
    {
        'name': 'Baseline (ReLU, lr=0.001, dropout=0.7)',
        'lr': 0.001, 'dropout': 0.7, 'activation': F.relu, 'seed': 16
    },
    # --- ADDED: New and Different Ablation Studies ---
    {
        'name': 'Fewer Hidden Units (64)',
        'hidden': 64
    },
    {
        'name': 'Lower Weight Decay (5e-5)',
        'weight_decay': 5e-5
    },
    {
        'name': 'Optimizer = SGD (lr=0.05)',  # SGD often requires a different learning rate
        'optimizer_name': 'sgd',
        'lr': 0.05
    },
    {
        'name': '3-Layer GCN Model',
        'model_class': GCN3Layer
    }
]

# --- Step 4: Run all experiments and print results ---
if __name__ == "__main__":
    results_list = []
    for i, params in enumerate(experiments):
        print(f"\n--- Running Experiment {i + 1}/{len(experiments)}: {params['name']} ---")
        try:
            result = run_experiment(params)
            results_list.append(result)
        except Exception as e:
            print(f"!!!!!! Experiment '{params['name']}' failed with error: {e} !!!!!!")

    print("\n\n--- Ablation Study Results ---")
    df = pd.DataFrame(results_list)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    print(df.to_string(index=False))