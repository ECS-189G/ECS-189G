import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import math
import random


# =========================================================================================
# 1. GRAPH CONVOLUTION LAYER
# =========================================================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# =========================================================================================
# 2. BASE CLASSES & PLACEHOLDERS
# =========================================================================================
class method:
    def __init__(self, mName, mDescription):
        self.mName = mName
        self.mDescription = mDescription


class evaluate:
    def __init__(self, eName, eDescription):
        self.eName = eName
        self.eDescription = eDescription


class setting:
    def __init__(self, sName, sDescription):
        self.sName = sName
        self.sDescription = sDescription


class Result:
    data = None

    def __init__(self, rName=None, rDescription=None):
        self.rName = rName
        self.rDescription = rDescription

    def save(self):
        pass


# =========================================================================================
# 3. DATASET LOADER
# =========================================================================================
class Dataset_Loader:
    def __init__(self, dName=None, path="../data/"):
        self.dataset_name = dName
        self.dataset_source_folder_path = path
        self.dataset_source_file_name = dName

    def adj_normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        print('Loading {} dataset...'.format(self.dataset_name))
        path_to_node_file = self.dataset_source_folder_path + self.dataset_source_file_name + "/node"
        path_to_link_file = self.dataset_source_folder_path + self.dataset_source_file_name + "/link"

        idx_features_labels = np.genfromtxt(path_to_node_file, dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(path_to_link_file, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
            edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'X': features, 'y': labels, 'utility': {'A': adj}}
        return {'graph': graph, 'train_test_val': train_test_val}


# =========================================================================================
# 4. EVALUATION METRICS
# =========================================================================================
class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        true_y = self.data['true_y'].cpu()
        pred_y = self.data['pred_y'].cpu()

        return ('Accuracy: ' + str(accuracy_score(true_y, pred_y)) + '\n'
                + 'Precision: ' + str(precision_score(true_y, pred_y, average='macro', zero_division=0)) + '\n'
                + 'Recall: ' + str(recall_score(true_y, pred_y, average='macro', zero_division=0)) + '\n'
                + 'F1: ' + str(f1_score(true_y, pred_y, average='macro', zero_division=0)))


# =========================================================================================
# 5. CONFIGURABLE GCN METHOD (MODIFIED TO BE MORE FLEXIBLE)
# =========================================================================================
class Configurable_Cora_GCN(method, nn.Module):
    data = None

    def __init__(self, mName, mDescription, nfeat, nclass, config):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Load hyperparameters from the configuration dictionary
        self.num_layers = config.get('num_layers', 2)  # Default to 2 layers
        self.hidden1 = config.get('hidden1', 100)
        self.hidden2 = config.get('hidden2', 100)
        self.hidden3 = config.get('hidden3', 50)  # For 3-layer model
        self.dropout = config.get('dropout', 0.5)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.max_epoch = config.get('max_epoch', 80)
        activation_function = config.get('activation', nn.ReLU)
        output_activation = config.get('output_activation', nn.LogSoftmax)
        self.loss_function_class = config.get('loss_function', nn.NLLLoss)

        # --- Build layers based on configuration ---
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Input Layer
        self.layers.append(GraphConvolution(nfeat, self.hidden1))
        self.activations.append(activation_function())

        # Hidden Layers
        if self.num_layers == 2:
            self.layers.append(GraphConvolution(self.hidden1, self.hidden2))
            self.activations.append(activation_function())
            final_hidden_size = self.hidden2
        elif self.num_layers == 3:
            self.layers.append(GraphConvolution(self.hidden1, self.hidden2))
            self.activations.append(activation_function())
            self.layers.append(GraphConvolution(self.hidden2, self.hidden3))
            self.activations.append(activation_function())
            final_hidden_size = self.hidden3
        else:  # 1 Layer Model
            final_hidden_size = self.hidden1

        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc_final = nn.Linear(final_hidden_size, nclass)
        self.output_func = output_activation(dim=1)

    def forward(self, x, adj):
        h = x
        # Pass through GCN layers and activations
        for i, layer in enumerate(self.layers):
            h = self.activations[i](layer(h, adj))
            h = self.dropout_layer(h)

        # Final fully connected layer
        output = self.output_func(self.fc_final(h))
        return output

    def train_loop(self, features, labels, adj, idx_train):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = self.loss_function_class()
        for epoch in range(self.max_epoch):
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, adj)
            train_loss = loss_function(output[idx_train], labels[idx_train])
            train_loss.backward()
            optimizer.step()

    def test(self, features, adj, idx_test):
        self.eval()
        with torch.no_grad():
            output = self.forward(features, adj)
        return output[idx_test].max(1)[1]

    def run(self):
        features = self.data['graph']['X']
        labels = self.data['graph']['y']
        adj = self.data['graph']['utility']['A']
        idx_train = self.data['train_test_val']['idx_train']
        idx_test = self.data['train_test_val']['idx_test']
        self.train_loop(features, labels, adj, idx_train)
        output = self.test(features, adj, idx_test)
        return {'pred_y': output, 'true_y': labels[idx_test]}


# =========================================================================================
# 6. EXPERIMENT SETTING
# =========================================================================================
class Cora_Setting(setting):
    def run_and_evaluate(self):
        learned_result = self.method.run()
        self.evaluate.data = learned_result
        return self.evaluate.evaluate()


# =========================================================================================
# 7. MAIN ABLATION STUDY EXECUTION
# =========================================================================================
if __name__ == '__main__':
    # --- Ablation Study Configurations (WITH NEW COMPARISONS) ---
    ablation_configs = {
        "Baseline (2-layer ReLU)": {},
        "Dropout = 0.0": {
            "dropout": 0.0
        },
        "Learning rate = 5e-2": {
            "learning_rate": 5e-2
        },
        "Output activation = Softmax, Loss = CrossEntropy": {
            "output_activation": nn.Softmax,
            "loss_function": nn.CrossEntropyLoss
        },
        "GC1/GC2 output size = 50": {
            "hidden1": 50,
            "hidden2": 50
        },
        "Epochs = 40": {
            "max_epoch": 40
        },
        "Activation = Sigmoid": {
            "activation": nn.Sigmoid
        },
        # --- NEWLY ADDED EXPERIMENTS ---
        "No Weight Decay (L2 Regularization)": {
            "weight_decay": 0.0
        },
        "Activation = LeakyReLU": {
            "activation": nn.LeakyReLU
        },
        "Shallower Model (1 GCN layer)": {
            "num_layers": 1
        },
        "Deeper Model (3 GCN layers)": {
            "num_layers": 3
        }
    }

    # --- Load Data Once ---
    # Path is unchanged as requested
    dataset_loader = Dataset_Loader(dName='cora', path='../../../data/stage_5_data/')
    # =====================================================================================
    loaded_data = dataset_loader.load()
    nfeat = loaded_data['graph']['X'].shape[1]
    nclass = loaded_data['graph']['y'].max().item() + 1

    # --- Run Experiments ---
    results_list = []
    for name, config in ablation_configs.items():
        print(f"--- Running Ablation: {name} ---")

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        method_instance = Configurable_Cora_GCN(
            mName=f'GCN_{name}',
            mDescription=f'GCN with config: {name}',
            nfeat=nfeat,
            nclass=nclass,
            config=config
        )
        method_instance.data = loaded_data

        setting_instance = Cora_Setting(
            sName=f'Cora_Experiment_{name}',
            sDescription=f'Running experiment: {name}'
        )
        setting_instance.method = method_instance
        setting_instance.evaluate = Evaluate_Metrics('multi-metric-eval', '')

        metrics_string = setting_instance.run_and_evaluate()

        metrics = {}
        for line in metrics_string.split('\n'):
            if ':' in line:
                key, value = line.split(':')
                metrics[key.strip()] = float(value.strip())

        results_list.append({
            "Changes Made": name,
            "Accuracy": metrics.get("Accuracy", "N/A"),
            "Precision": metrics.get("Precision", "N/A"),
            "Recall": metrics.get("Recall", "N/A"),
            "F1 Score": metrics.get("F1", "N/A")
        })

    # --- Display Final Results Table ---
    df = pd.DataFrame(results_list)
    print("\n--- Ablation Study Results ---")
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        df[col] = df[col].map('{:.3f}'.format)
    print(df.to_string(index=False))