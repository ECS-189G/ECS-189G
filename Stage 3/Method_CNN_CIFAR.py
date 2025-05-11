from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
# For LR Scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    max_epoch = 10  # As per user's current run
    learning_rate = 2e-3
    mps_device = None

    mean_norm = [0.4914, 0.4822, 0.4465]  # CIFAR-10 specific mean
    std_norm = [0.2470, 0.2435, 0.2616]  # CIFAR-10 specific std

    def __init__(self, mName, mDescription, mps=False):
        method.__init__(self, mName, mDescription)

        if mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.mps_device = self.device
            print(f'Using MPS device: {self.device}')
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f'Using CUDA device: {self.device}')
        else:
            self.device = torch.device("cpu")
            print(f'Using CPU device. WARNING: Training may be very slow.')

        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.norm2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.norm3 = nn.BatchNorm2d(256)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.act7 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.act8 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)

        self.to(self.device)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        x = self.norm3(x)

        x = self.flat(x)
        x = self.act7(self.fc1(x))
        x = self.act8(self.fc2(x))
        y_pred_logits = self.fc3(x)
        return y_pred_logits

    def _preprocess_data(self, X_numpy, is_train=True):  # is_train currently not used for different transforms
        if isinstance(X_numpy, list):
            X_numpy = np.array(X_numpy)

        X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
        if X_tensor.max() > 1.1:
            X_tensor = X_tensor / 255.0

        normalize_transform = transforms.Normalize(self.mean_norm, self.std_norm)
        X_tensor_normalized = normalize_transform(X_tensor)

        return X_tensor_normalized


    def _execute_training_loop(self, X_train_raw, y_train_raw):
        # CORRECTED: Calls nn.Module.train(True) to set training mode
        # This call here is optional if the one inside epoch loop is present,
        # but good for clarity that the whole method is for training.
        self.train()

        print(f"Input data X shape: {np.array(X_train_raw).shape}")
        print(f"Number of training samples: {len(y_train_raw)}")
        print(f"Unique labels in y: {set(y_train_raw)}")

        X_tensor = self._preprocess_data(X_train_raw, is_train=True)
        y_tensor = torch.tensor(y_train_raw, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=1e-6)
        loss_function = nn.CrossEntropyLoss()

        epoch_losses = []
        epoch_accuracies = []

        print(f"Starting training for {self.max_epoch} epochs...")

        for epoch in range(self.max_epoch):
            self.train()  # CORRECTED: Calls nn.Module.train(True)
            batch_total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for data_batch, target_batch in loader:
                data_batch, target_batch = data_batch.to(self.device), target_batch.to(self.device)

                optimizer.zero_grad()
                y_pred_logits = self.forward(data_batch)
                loss = loss_function(y_pred_logits, target_batch)

                loss.backward()
                optimizer.step()

                batch_total_loss += loss.item() * data_batch.size(0)
                _, predicted_labels = torch.max(y_pred_logits, 1)
                correct_predictions += (predicted_labels == target_batch).sum().item()
                total_samples += target_batch.size(0)

            scheduler.step()

            avg_epoch_loss = batch_total_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            epoch_losses.append(avg_epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f'Epoch [{epoch + 1}/{self.max_epoch}] | LR: {current_lr:.2e} | Loss: {avg_epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}')

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Average Loss')
        plt.title('Training Loss')

        plt.subplot(1, 2, 2)
        plt.plot(epoch_accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')

        plt.tight_layout()
        import os
        os.makedirs('../../result/stage_3_result/', exist_ok=True)
        plt.savefig('../../result/stage_3_result/cifar_train_metrics.png')
        plt.close()
        print("Training finished. Metrics plot saved.")

    def test(self, X_test_raw):
        self.eval()  # CORRECTED: Calls nn.Module.train(False)

        print(f"Test data X shape: {np.array(X_test_raw).shape}")
        X_tensor = self._preprocess_data(X_test_raw, is_train=False)

        test_dataset = TensorDataset(X_tensor)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        all_predictions = []
        with torch.no_grad():
            for data_batch_tuple in test_loader:
                data_batch = data_batch_tuple[0].to(self.device)
                output_logits = self.forward(data_batch)
                _, predicted_labels = torch.max(output_logits, 1)
                all_predictions.extend(predicted_labels.cpu().numpy())
        return np.array(all_predictions)

    def run(self):
        print('Method_CNN_CIFAR running...')
        if self.mps_device and self.device != self.mps_device:
            self.to(self.mps_device)
            print(f"Model explicitly moved to {self.mps_device} in run method.")

        if self.data is None or 'train' not in self.data or 'X' not in self.data['train'] or \
                'y' not in self.data['train']:
            print("Error: Training data not loaded correctly.")
            return {'pred_y': [], 'true_y': []}

        print('--start training...')
        # UPDATED CALL to the renamed training loop method
        self._execute_training_loop(self.data['train']['X'], self.data['train']['y'])

        if self.data is None or 'test' not in self.data or 'X' not in self.data['test'] or \
                'y' not in self.data['test']:
            print("Error: Testing data not loaded correctly.")
            return {'pred_y': [], 'true_y': []}

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y.tolist(), 'true_y': self.data['test']['y']}