from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms  # For normalization

class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3

    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    def __init__(self, mName, mDescription, mps_override=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        if mps_override and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device (override).")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device.")

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.act3 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

        self.data_transform = transforms.Normalize(self.mnist_mean, self.mnist_std)
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

        x = self.fc3(x)
        x = self.act3(x)
        x = self.drop1(x)

        y_pred_logits = self.fc4(x)
        return y_pred_logits

    def _preprocess_data(self, X_numpy):
        X_tensor = torch.FloatTensor(np.array(X_numpy))
        if X_tensor.max() > 1.1:
            X_tensor = X_tensor / 255.0
        X_tensor = self.data_transform(X_tensor)
        return X_tensor

    # RENAMED the training loop method
    def _execute_training_loop(self, X_train_raw, y_train_raw):
        self.train()  # CORRECTED: Calls nn.Module.train(True) to set training mode

        X_tensor = self._preprocess_data(X_train_raw)
        y_true_tensor = torch.LongTensor(np.array(y_train_raw))

        train_dataset = TensorDataset(X_tensor, y_true_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        loss_function = nn.CrossEntropyLoss()

        epoch_losses = []
        print(f"Starting training for {self.max_epoch} epochs on {self.device}...")

        for epoch in range(self.max_epoch):
            self.train()  # Ensure model is in training mode at the start of each epoch
            batch_loss_sum = 0.0
            correct_preds_sum = 0
            total_samples = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                y_pred_logits = self.forward(X_batch)
                loss = loss_function(y_pred_logits, y_batch)

                loss.backward()
                optimizer.step()

                batch_loss_sum += loss.item() * X_batch.size(0)
                correct_preds_sum += (y_pred_logits.argmax(dim=1) == y_batch).sum().item()
                total_samples += y_batch.size(0)

            avg_epoch_loss = batch_loss_sum / total_samples
            epoch_accuracy = correct_preds_sum / total_samples
            epoch_losses.append(avg_epoch_loss)

            print(
                f'Epoch: {epoch + 1:02d}/{self.max_epoch} | Loss: {avg_epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}')

        plt.figure()
        plt.plot(epoch_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.title('Training Loss over Epochs')
        import os
        os.makedirs('../../result/stage_3_result/', exist_ok=True)  # Ensure directory exists
        plt.savefig('../../result/stage_3_result/mnist_loss.png')
        plt.close()
        print("Training finished. Loss plot saved.")

    def test(self, X_test_raw):
        self.eval()  # CORRECTED: Calls nn.Module.train(False) to set evaluation mode

        X_tensor = self._preprocess_data(X_test_raw)

        test_dataset = TensorDataset(X_tensor)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        all_predictions = []
        with torch.no_grad():
            for X_batch_tuple in test_loader:
                X_batch = X_batch_tuple[0].to(self.device)
                y_pred_logits = self.forward(X_batch)
                all_predictions.extend(y_pred_logits.argmax(dim=1).cpu().tolist())

        return all_predictions

    def run(self):
        print('Method_CNN_MNIST running...')
        if self.data is None or 'train' not in self.data or 'test' not in self.data \
                or 'X' not in self.data['train'] or 'y' not in self.data['train'] \
                or 'X' not in self.data['test'] or 'y' not in self.data['test']:
            print("Error: Data not found or not in the expected dictionary format.")
            return {'pred_y': [], 'true_y': []}

        print('--start training...')
        # UPDATED CALL to the renamed training loop method
        self._execute_training_loop(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}