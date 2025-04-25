""""
Concrete MethodModule for a multi-layer perceptron (MLP) with visualization.
"""
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from local_code.base_class.method import method as BaseMethod


class Method_MLP(BaseMethod, nn.Module):
    """Two-hidden-layer MLP with training curve visualization."""
    data = None
    max_epoch = 150          # maximum number of training epochs
    learning_rate = 1e-3     # optimizer learning rate

    def __init__(self, mName, mDescription):
        """Initialize base method metadata and network layers."""
        BaseMethod.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Define network architecture
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """Forward pass through hidden layers and output layer."""
        x = self.fc1(x)
        if x.dim() > 1 and x.size(0) > 1:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if x.dim() > 1 and x.size(0) > 1:
            x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout(x)

        return self.fc3(x)

    def fit(self, X, y):
        """Train the model and generate training curves."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        # Convert data to tensors
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))

        losses = []
        accuracies = []

        # Training loop without additional checks
        for epoch in range(self.max_epoch):
            self.train()
            logits = self.forward(X_tensor)
            loss = loss_fn(logits, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()
                preds = torch.softmax(logits, dim=1).argmax(dim=1)
                accuracy = (preds == y_tensor).float().mean().item()

            losses.append(loss.item())
            accuracies.append(accuracy)

            # Log progress every 2 epochs
            if epoch % 2 == 0:
                print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {loss.item():.6f}')

        # Plot training loss and accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(accuracies)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy over Epochs')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('../../result/stage_2_result/training_curves.png')
        plt.close()

    def test(self, X):
        """Test the model and return predicted labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.FloatTensor(np.array(X)))
            preds = torch.softmax(logits, dim=1).argmax(dim=1)
        return preds

    def run(self):
        """Preprocess data, train, and test the model."""
        print('method running...')
        print('preprocessing data...')
        self.data['train']['X'] = np.array(self.data['train']['X'], dtype=np.float32) / 255.0
        self.data['test']['X'] = np.array(self.data['test']['X'], dtype=np.float32) / 255.0

        print('start training...')
        self.fit(self.data['train']['X'], self.data['train']['y'])

        print('start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y.tolist(), 'true_y': self.data['test']['y']}


