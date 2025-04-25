
"""
Ablation study: compare variants of the MLP architecture and training settings.
"""
import sys
import os

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(__file__, '../../..'))
sys.path.insert(0, project_root)

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader

# ---- Load CSV data ----
def load_csv_data():
    base_dir = os.path.abspath(os.path.join(__file__, '../../..'))
    data_dir = os.path.join(base_dir, 'data', 'stage_2_data') + os.sep

    train_data = Dataset_Loader('train', '')
    train_data.dataset_source_folder_path = data_dir
    train_data.dataset_source_file_name = 'train.csv'

    test_data = Dataset_Loader('test', '')
    test_data.dataset_source_folder_path = data_dir
    test_data.dataset_source_file_name = 'test.csv'

    tr = train_data.load()
    te = test_data.load()
    X_train = np.array(tr['X'], dtype=np.float32)
    y_train = np.array(tr['y'], dtype=int)
    X_test  = np.array(te['X'], dtype=np.float32)
    y_test  = np.array(te['y'], dtype=int)
    return X_train, y_train, X_test, y_test

# ---- Flexible MLP builder ----
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation='leaky'):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            if activation == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.1))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---- Train & evaluate function ----
def train_and_eval(cfg, epochs=30):
    X_train, y_train, X_test, y_test = load_csv_data()
    model = SimpleMLP(
        input_dim=784,
        hidden_dims=cfg['hidden'],
        dropout=cfg['dropout'],
        activation=cfg['activation']
    )
    # choose optimizer
    if cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg['lr'],
            momentum=cfg.get('momentum', 0)
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0)
        )
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.FloatTensor(X_train)
    ytr = torch.LongTensor(y_train)

    for _ in range(epochs):
        model.train()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).argmax(1).numpy()

    return {
        'Accuracy':  accuracy_score( y_test, preds),
        'Precision': precision_score(y_test, preds, average='macro', zero_division=0),
        'Recall':    recall_score(   y_test, preds, average='macro'),
        'F1 Score':  f1_score(       y_test, preds, average='macro'),
    }

# ---- Configurations for ablation ----
configs = [
    { 'name': 'Baseline (256→128, LeakyReLU, lr=1e-3, AdamW, wd=1e-4)',
      'hidden': [256, 128], 'dropout': 0.3,
      'activation': 'leaky', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 1e-4 },

    { 'name': 'Remove dropout (p=0)',
      'hidden': [256, 128], 'dropout': 0.0,
      'activation': 'leaky', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 1e-4 },

    { 'name': 'Replace LeakyReLU with ReLU',
      'hidden': [256, 128], 'dropout': 0.3,
      'activation': 'relu', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 1e-4 },

    { 'name': 'Increase hidden units (512→256)',
      'hidden': [512, 256], 'dropout': 0.3,
      'activation': 'leaky', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 1e-4 },

    { 'name': 'Optimizer = SGD (lr=1e-3, momentum=0.9)',
      'hidden': [256, 128], 'dropout': 0.3,
      'activation': 'leaky', 'optimizer': 'sgd', 'lr': 1e-3, 'momentum': 0.9 },

    { 'name': 'Remove weight decay (wd=0)',
      'hidden': [256, 128], 'dropout': 0.3,
      'activation': 'leaky', 'optimizer': 'adamw', 'lr': 1e-3, 'weight_decay': 0.0 },
]

# ---- Run ablation experiments ----
results = []
for cfg in configs:
    print(f"Running config: {cfg['name']}")
    metrics = train_and_eval(cfg, epochs=30)
    metrics['Configuration'] = cfg['name']
    results.append(metrics)

# ---- Collect & save results ----
df = pd.DataFrame(results)[['Configuration','Accuracy','Precision','Recall','F1 Score']]
print(df.to_string(index=False))
df.to_csv('ablation_results.csv', index=False)
print('Saved ablation_results.csv')

# ---- Plot accuracy comparison ----
plt.figure(figsize=(10,5))
plt.bar(df['Configuration'], df['Accuracy'])
plt.xlabel('Experiment Variation')
plt.ylabel('Accuracy')
plt.title('Ablation Study: Test Accuracy per Variation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ablation_accuracy.png')
print('Saved ablation_accuracy.png')
