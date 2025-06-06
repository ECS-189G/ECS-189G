import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

# These imports assume the original project structure is maintained.
from local_code.stage_5_code.pubmed_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.pubmed_code.train import train
from local_code.stage_5_code.pubmed_code.layers import GraphConvolution


# --- Step 1: Define Model Architectures Locally ---

class GCN2Layer(nn.Module):
    """Standard 2-Layer GCN Model."""

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
    """A Deeper 3-Layer GCN Model."""

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


# --- Step 2: Define the Main Experiment Runner Function ---
def run_experiment(params, dataset_name):
    """
    Runs a full training and testing experiment for a given dataset.
    """
    print(f"Loading {dataset_name} dataset...")
    seed = params.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize data loader
    curr = Dataset_Loader()
    curr.dataset_name = dataset_name
    curr.dataset_source_folder_path = f"../../../data/stage_5_data/{dataset_name}/"
    adj, features, labels, idx_train, idx_val, idx_test = curr.load()

    # Initialize model based on parameters
    model_class = params.get('model_class', GCN2Layer)
    model = model_class(nfeat=features.shape[1],
                        nhid=params.get('hidden', 128),
                        nclass=labels.max().item() + 1,
                        dropout=params.get('dropout', 0.6),
                        activation=params.get('activation', F.relu))

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=params.get('lr', 0.001),
                           weight_decay=params.get('weight_decay', 5e-4))

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop with early stopping
    for epoch in range(params.get('epochs', 200)):
        loss = []  # Dummy list for train function compatibility
        val_loss = train(epoch, adj, features, labels, idx_train, idx_val, idx_test, model, optimizer,
                         argparse.Namespace(fastmode=False), loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if params.get('epochs', 200) > epoch + 1:
                print(f"Early stopping at epoch {epoch + 1} for experiment '{params['name']}'")
            break

    # Load best model and get test metrics
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

    # Return results in a dictionary
    return {
        "Change Made": params['name'],
        "Accuracy": acc,
        "Other Metrics": f"Precision= {precision:.4f}\nRecall = {recall:.4f}\nF1 Score = {f1:.4f}"
    }


# --- Step 3: Define the Ablation Experiment Suite ---
# This list contains 1 baseline and 6 different ablation experiments.
ablation_suite = [
    # This is our baseline model for comparison
    {
        'name': 'Baseline (2-Layer, ReLU, lr=0.001, dropout=0.6)',
        'lr': 0.001, 'dropout': 0.6, 'hidden': 128, 'model_class': GCN2Layer,
        'activation': F.relu, 'weight_decay': 5e-4
    },

    # --- 6 Different Ablation Experiments ---
    # 1. Model Depth
    {
        'name': 'Ablation 1: Deeper Model (3-Layer)',
        'model_class': GCN3Layer
    },
    # 2. Activation Function
    {
        'name': 'Ablation 2: Activation = LeakyReLU',
        'activation': F.leaky_relu
    },
    # 3. Dropout Rate
    {
        'name': 'Ablation 3: Lower Dropout (0.5)',
        'dropout': 0.5
    },
    # 4. Weight Decay (L2 Regularization)
    {
        'name': 'Ablation 4: Lower Weight Decay (5e-5)',
        'weight_decay': 5e-5
    },
    # 5. Learning Rate
    {
        'name': 'Ablation 5: Higher Learning Rate (0.01)',
        'lr': 0.01
    },
    # 6. Hidden Layer Size
    {
        'name': 'Ablation 6: Larger Hidden Layer (256)',
        'hidden': 256
    }
]

# --- Step 4: Run the Ablation Suite for PubMed ---
if __name__ == "__main__":
    results_list = []
    baseline_params = ablation_suite[0]  # Use the first entry as the default configuration

    print("==============================================")
    print("  RUNNING ABLATION SUITE ON: PubMed           ")
    print("==============================================")

    # Loop through each experiment
    for i, experiment_changes in enumerate(ablation_suite):
        # Start with baseline params and overwrite with specific experiment changes
        run_params = baseline_params.copy()
        run_params.update(experiment_changes)

        print(f"\n--- Running Experiment {i + 1}/{len(ablation_suite)}: {run_params['name']} ---")
        try:
            result = run_experiment(run_params, "pubmed")
            results_list.append(result)
        except Exception as e:
            print(f"!!!!!! Experiment '{run_params['name']}' failed with error: {e} !!!!!!")

    # Create and print the final results table
    print("\n\n--- Final PubMed Ablation Study Results ---")
    df = pd.DataFrame(results_list)

    # Set pandas display options to show the full table
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    print(df.to_string(index=False))