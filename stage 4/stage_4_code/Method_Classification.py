import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import traceback
import time
from tqdm import tqdm
from sklearn.metrics import classification_report


class method:  # Placeholder for your base class
    def __init__(self, mName, mDescription):
        self.method_name = mName
        self.method_description = mDescription

    def run(self, *args, **kwargs):
        raise NotImplementedError


class Method_Classification(nn.Module, method):
    def __init__(self, mName: str, mDescription: str,
                 embedding_dim: int = 100,
                 hidden_dim: int = 128,
                 num_lstm_layers: int = 1,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 max_epochs: int = 8,
                 grad_clip: float = 3.0,
                 dropout_rate: float = 0.3,
                 weight_decay: float = 1e-5,
                 val_split_percent: float = 0.1,
                 patience: int = 3,
                 seed: int = 42,
                 device: str = None
                 ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[{self.method_name}] Using device: {self.device}")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.val_split_percent = val_split_percent
        self.patience = patience
        self.seed = seed

        self.embedding = None
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_lstm_layers > 1 else 0
        )
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1),
            nn.Softmax(dim=1)
        )

        # Assuming the complex head structure
        self.fc1 = nn.Linear(self.hidden_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()

        self.fc_intermediate = nn.Linear(256, 128)
        self.bn_intermediate = nn.BatchNorm1d(128)
        self.relu_intermediate = nn.ReLU()

        self.fc2 = nn.Linear(128, 1)

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.data = None
        self.to(self.device)
        print(f"[{self.method_name}] Model initialized and moved to {self.device}.")

    def initialize_embeddings(self, embedding_matrix: torch.Tensor):
        if embedding_matrix.shape[1] != self.embedding_dim:
            print(f"Warning: Configured embedding_dim ({self.embedding_dim}) "
                  f"does not match embedding_matrix dimension ({embedding_matrix.shape[1]}). "
                  f"Updating model's embedding_dim to match matrix.")
            self.embedding_dim = embedding_matrix.shape[1]
            self.lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.dropout_rate if self.num_lstm_layers > 1 else 0
            )
            self.to(self.device)

        self.embedding = nn.Embedding(
            num_embeddings=embedding_matrix.shape[0],
            embedding_dim=embedding_matrix.shape[1]
        )
        self.embedding.weight.data.copy_(embedding_matrix)
        # --- MODIFICATION FOR FROZEN GLOVE ---
        self.embedding.weight.requires_grad = False  # Embeddings will NOT be updated during training
        # --- END MODIFICATION ---
        self.embedding.to(self.device)
        status = "fine-tunable" if self.embedding.weight.requires_grad else "frozen"
        print(
            f"[{self.method_name}] Embedding layer initialized with {status} GloVe vectors of dim: {self.embedding_dim}")

    def forward(self, x: torch.Tensor):
        if self.embedding is None:
            raise RuntimeError("Embedding layer not initialized. Call initialize_embeddings() first.")
        x_embedded = self.embedding(x)
        output, _ = self.lstm(x_embedded)
        attn_weights = self.attention(output)
        context = torch.bmm(attn_weights.transpose(1, 2), output).squeeze(1)

        context_after_initial_dropout = self.dropout_layer(context)

        x = self.fc1(context_after_initial_dropout)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout_layer(x)

        x = self.fc_intermediate(x)
        x = self.bn_intermediate(x)
        x = self.relu_intermediate(x)
        x = self.dropout_layer(x)

        x = self.fc2(x)
        return x.squeeze()

    def create_validation_set(self, train_dataset: Dataset):
        train_size = int((1 - self.val_split_percent) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        print(f"[{self.method_name}] Split training data: Train subset size: {len(train_subset)}, "
              f"Validation subset size: {len(val_subset)}")
        return train_subset, val_subset

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader = None):
        print(f"[{self.method_name}] Starting model training for {self.max_epochs} epochs...")
        self.train()

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.parameters(),
            # Note: If embeddings are frozen, they won't be in self.parameters() with requires_grad=True
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler_patience = max(1, self.patience // 2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=scheduler_patience
        )
        print(f"[{self.method_name}] Learning rate scheduler initialized (patience={scheduler_patience}).")
        print(f"[{self.method_name}] Early stopping enabled with patience={self.patience}.")

        wait = 0
        best_loss_val = float('inf')

        for epoch in range(self.max_epochs):
            self.train()
            train_loss = 0.0
            train_acc = 0.0
            num_train_batches = 0

            train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs} [Train]")
            for feature, target in train_loop:
                feature, target = feature.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out = self.forward(feature)
                loss = self.criterion(out, target.float())
                pred = (torch.sigmoid(out) > 0.5).long()
                acc = (pred == target).float().mean()
                train_loss += loss.item()
                train_acc += acc.item()
                num_train_batches += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                self.optimizer.step()
                train_loop.set_postfix(loss=loss.item(), acc=acc.item())

            avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0.0
            avg_train_acc = train_acc / num_train_batches if num_train_batches > 0 else 0.0
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            print(f"[{self.method_name}] Epoch {epoch + 1}/{self.max_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")

            if val_loader:
                val_loss, val_acc = self.evaluate_model(val_loader, stage="Validation")
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                print(f"[{self.method_name}] Epoch {epoch + 1}/{self.max_epochs}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                scheduler.step(val_loss)

                if val_loss < best_loss_val:
                    best_loss_val = val_loss
                    self.best_model_state = self.state_dict().copy()
                    self.best_val_acc = val_acc
                    wait = 0
                    print(f"[{self.method_name}] New best model saved based on validation loss: {best_loss_val:.4f}")
                else:
                    wait += 1
                    print(f"[{self.method_name}] No validation loss improvement for {wait}/{self.patience} epochs.")
                    if wait >= self.patience:
                        print(f"[{self.method_name}] Early stopping triggered after {epoch + 1} epochs.")
                        break

        if self.best_model_state:
            self.load_state_dict(self.best_model_state)
            print(
                f"[{self.method_name}] Loaded best model state from epoch with validation loss: {best_loss_val:.4f} and acc: {self.best_val_acc:.4f}")
        else:
            print(f"[{self.method_name}] Training finished. No best model state loaded.")

    def evaluate_model(self, data_loader: DataLoader, stage: str = "Evaluation"):
        self.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()

        eval_loop = tqdm(data_loader, desc=f"[{self.method_name}] {stage}")
        with torch.no_grad():
            for i, (feature, target) in enumerate(eval_loop):
                feature, target = feature.to(self.device), target.to(self.device)
                try:
                    out = self.forward(feature)
                    loss = self.criterion(out, target.float())
                    pred = (torch.sigmoid(out) > 0.5).long()
                    acc = (pred == target).float().mean()
                    total_loss += loss.item()
                    total_acc += acc.item()
                    num_batches += 1
                except Exception as e:
                    print(f"[{self.method_name}] Error in {stage} batch {i}: {e}")
                    traceback.print_exc()
                    continue
        if num_batches == 0:
            print(f"[{self.method_name}] No batches processed in {stage}.")
            return float('inf'), 0.0

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        if stage != "Validation":
            print(f"[{self.method_name}] {stage} Results - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        return avg_loss, avg_acc

    def test_model(self, test_loader: DataLoader):
        self.eval()
        all_target = []
        all_predicted = []
        all_probabilities = []
        if not hasattr(self, 'criterion') or self.criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()

        print(f"[{self.method_name}] Testing model on test set...")
        avg_test_loss, avg_test_acc = self.evaluate_model(test_loader, stage="Test")
        print(f'[{self.method_name}] Test Summary - Avg Loss: {avg_test_loss:.4f}, Avg Acc: {avg_test_acc:.4f}')

        with torch.no_grad():
            for feature, target in tqdm(test_loader, desc=f"[{self.method_name}] Generating Test Predictions"):
                feature, target = feature.to(self.device), target.to(self.device)
                out = self.forward(feature)
                probabilities = torch.sigmoid(out)
                pred = (probabilities > 0.5).long()
                all_target.extend(target.cpu().numpy())
                all_predicted.extend(pred.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        if all_target and all_predicted:
            print(f"[{self.method_name}] Classification Report:")
            print(classification_report(all_target, all_predicted, zero_division=0))
        else:
            print(f"[{self.method_name}] No samples to generate classification report for.")

        return {'pred_y': all_predicted, 'true_y': all_target, 'probabilities': all_probabilities}

    def run(self, data_dict: dict):
        self.data = data_dict
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        print(f"[{self.method_name}] Random seeds set to {self.seed} for this run.")
        print(
            f"[{self.method_name}] Train label distribution: {torch.unique(self.data['train']['y'], return_counts=True)}")
        print(
            f"[{self.method_name}] Test label distribution: {torch.unique(self.data['test']['y'], return_counts=True)}")
        print(
            f"[{self.method_name}] Size of train X: {self.data['train']['X'].shape}, y: {self.data['train']['y'].shape}")
        print(f"[{self.method_name}] Size of test X: {self.data['test']['X'].shape}, y: {self.data['test']['y'].shape}")

        self.initialize_embeddings(self.data['embedding'])
        train_full_dataset = torch.utils.data.TensorDataset(self.data['train']['X'], self.data['train']['y'])
        test_dataset = torch.utils.data.TensorDataset(self.data['test']['X'], self.data['test']['y'])
        train_subset, val_subset = self.create_validation_set(train_full_dataset)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        start_time = time.time()
        self.train_model(train_loader, val_loader)
        training_time = time.time() - start_time
        print(f"[{self.method_name}] Training completed in {training_time:.2f} seconds")
        test_results = self.test_model(test_loader)
        try:
            self.plot_metrics()
        except Exception as e:
            print(f"[{self.method_name}] Error plotting metrics: {e}")
            traceback.print_exc()
        return {'pred_y': test_results['pred_y'], 'true_y': test_results['true_y']}

    def plot_metrics(self):
        if not self.history['train_loss']:
            print(f"[{self.method_name}] No training history to plot.")
            return

        plot_dir = '../../result/stage_4_result/'
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{self.method_name.replace(" ", "_")}_classification_metrics.png')

        N = len(self.history['train_loss'])
        epochs_range = list(range(1, N + 1))
        val_N = len(self.history['val_loss'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{self.method_name} Training and Validation Metrics')

        ax1.plot(epochs_range, self.history['train_loss'], marker='o', label='Training Loss')
        if self.history['val_loss']:
            val_epochs_range = list(range(1, val_N + 1))
            ax1.plot(val_epochs_range, self.history['val_loss'], marker='o', label='Validation Loss')
        ax1.set_xticks(epochs_range if N <= 20 else range(1, N + 1, N // 10 if N > 10 else 1))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss over Epochs')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.plot(epochs_range, self.history['train_acc'], marker='o', label='Training Accuracy')
        if self.history['val_acc']:
            val_epochs_range = list(range(1, val_N + 1))
            ax2.plot(val_epochs_range, self.history['val_acc'], marker='o', label='Validation Accuracy')
        ax2.set_xticks(epochs_range if N <= 20 else range(1, N + 1, N // 10 if N > 10 else 1))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"[{self.method_name}] Metrics plot saved to {plot_path}")