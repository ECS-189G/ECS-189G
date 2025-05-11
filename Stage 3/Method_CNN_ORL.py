from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

class Method_CNN_ORL(method, nn.Module):
    data = None
    max_epoch = 30
    learning_rate = 1e-3
    orl_mean = [0.45, 0.45, 0.45]
    orl_std = [0.25, 0.25, 0.25]

    def __init__(self, mName, mDescription, mps=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(in_features=64 * 28 * 23, out_features=256)
        self.act3 = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=256, out_features=40)

        self.normalize_transform = transforms.Normalize(mean=self.orl_mean, std=self.orl_std)

        self.device = torch.device("cpu")
        if mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"MPS device selected and available: {self.device}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"CUDA device selected and available: {self.device}")
        elif mps:
            print("MPS device selected but not available, using CPU.")
        else:
            print("No GPU (CUDA/MPS) available, using CPU.")

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flat(x)
        x = self.act3(self.fc3(x))
        x = self.drop(x)
        return self.fc4(x)

    def _preprocess_data(self, X_numpy):
        if not isinstance(X_numpy, np.ndarray):
            X_numpy = np.array(X_numpy)

        X_tensor = torch.from_numpy(X_numpy).float()
        if X_tensor.max() > 1.1:
            X_tensor /= 255.0
        X_tensor = self.normalize_transform(X_tensor)
        return X_tensor

    def _run_training_loop(self, X_train_raw, y_train_raw):
        X_tensor = self._preprocess_data(X_train_raw)
        y_tensor = torch.from_numpy(np.array(y_train_raw) - 1).long()  # 0-indexed labels

        train_dataset = TensorDataset(X_tensor, y_tensor)  # Use full X_tensor, y_tensor for dataset
        # This loader is for actual training steps, shuffle is good here
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)
        criterion = nn.CrossEntropyLoss()

        print(f"Starting training for {self.max_epoch} epochs on device: {self.device}")

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss = 0.0
            correct_preds_epoch = 0
            total_samples_epoch = 0

            for xb, yb in train_loader:  # Batches from shuffled train_loader
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                preds = self.forward(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)
                correct_preds_epoch += (preds.argmax(dim=1) == yb).sum().item()
                total_samples_epoch += yb.size(0)

            scheduler.step()

            avg_epoch_loss = epoch_loss / total_samples_epoch
            epoch_accuracy = correct_preds_epoch / total_samples_epoch
            current_lr = scheduler.get_last_lr()[0]
            print(
                f'Epoch {epoch + 1:03d}/{self.max_epoch} | LR: {current_lr:.2e} | Train Loss: {avg_epoch_loss:.4f} | Train Acc (in train mode): {epoch_accuracy:.4f}')

            # << --- MODIFIED EVALUATION BLOCK --- >>
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.max_epoch:
                self.eval()
                temp_evaluator = Evaluate_Accuracy('temp_eval_train_set_eval_mode', '')
                all_eval_preds_list = []

                # Create a NEW DataLoader for evaluation on training set, with shuffle=FALSE
                eval_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)

                with torch.no_grad():
                    for xb_eval, _ in eval_train_loader:  # Iterate through non-shuffled loader
                        xb_eval = xb_eval.to(self.device)
                        all_eval_preds_list.append(self.forward(xb_eval).argmax(dim=1).cpu())

                all_eval_preds_tensor = torch.cat(all_eval_preds_list)
                # y_tensor is the original, ordered, 0-indexed training labels.
                # Predictions from non-shuffled loader will match this order.
                temp_evaluator.data = {'true_y': y_tensor.cpu(), 'pred_y': all_eval_preds_tensor}
                acc_eval_mode = temp_evaluator.evaluate()
                print(f'Epoch {epoch + 1:03d} Eval Mode Train Acc (on full train set): {acc_eval_mode:.4f}')
                self.train()  # Switch back to train mode

        print("Training finished.")

    def test(self, X_test_raw):
        self.eval()
        X_tensor = self._preprocess_data(X_test_raw)

        test_ds = TensorDataset(X_tensor)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        all_preds_list = []
        with torch.no_grad():
            for xb_tuple in test_loader:
                xb = xb_tuple[0].to(self.device)
                preds_logits = self.forward(xb)
                all_preds_list.append(preds_logits.argmax(dim=1).cpu())

        final_preds_0_indexed = torch.cat(all_preds_list)
        return (final_preds_0_indexed + 1).tolist()  # Return 1-indexed predictions

    def run(self):
        print('Method_CNN_ORL running...')
        if self.data is None or 'train' not in self.data or 'test' not in self.data \
                or 'X' not in self.data['train'] or 'y' not in self.data['train'] \
                or 'X' not in self.data['test'] or 'y' not in self.data['test']:
            print("Error: Data not found or not in the expected dictionary format.")
            return {'pred_y': [], 'true_y': []}

        # Advise to check test set size and label format
        print(f"Test data X shape: {np.array(self.data['test']['X']).shape}")
        print(
            f"Test data y example (first 5, should be 1-indexed if pred_y is 1-indexed): {np.array(self.data['test']['y'])[:5]}")

        print('--start training...')
        self._run_training_loop(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])  # Returns 1-indexed predictions

        true_y_for_eval = self.data['test']['y']
        # Ensure true_y_for_eval is 1-indexed if pred_y is 1-indexed for the external evaluator
        # (Assuming it is for ORL standard labels 1-40)

        return {'pred_y': pred_y, 'true_y': true_y_for_eval}