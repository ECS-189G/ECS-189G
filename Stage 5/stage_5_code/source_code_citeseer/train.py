import time
import torch
import torch.nn.functional as F
from local_code.stage_5_code.pubmed_code.utils import accuracy
from local_code.stage_5_code.pubmed_code.models import GCN
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def train(epoch, adj, features, labels, idx_train, idx_val, idx_test, model, optimizer, args, loss):
    t = time.time()
    model.train()  #
    optimizer.zero_grad()  #
    output = model(features, adj)  #
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])  #
    acc_train = accuracy(output[idx_train], labels[idx_train])  #
    loss_train.backward()  #
    optimizer.step()  #

    if not args.fastmode:  #
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()  #
        output = model(features, adj)  #

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])  #
    acc_val = accuracy(output[idx_val], labels[idx_val])  #

    loss.append(loss_val.item())  #
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))  #

    # --- ADDED: Return validation loss for early stopping ---
    return loss_val.item()


def plot_learning_curves(epochs, val_losses):
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()


def test(adj, features, labels, idx_train, idx_val, idx_test, model):
    model.eval()  #
    output = model(features, adj)  #

    # Get predictions for the test set
    preds = output[idx_test].max(1)[1].cpu().numpy()
    labels_test = labels[idx_test].cpu().numpy()

    # Calculate metrics using scikit-learn for multi-class classification
    acc_test = accuracy(output[idx_test], labels[idx_test])  #
    precision_test = precision_score(labels_test, preds, average='macro')
    recall_test = recall_score(labels_test, preds, average='macro')
    f1_test = f1_score(labels_test, preds, average='macro')

    # Print in the desired format
    print("\nsaving results...")
    print("evaluating performance...")
    print("Accuracy: {:.5f}".format(acc_test.item()))
    print("Precision: {}".format(precision_test))
    print("Recall: {}".format(recall_test))
    print("F1: {}".format(f1_test))