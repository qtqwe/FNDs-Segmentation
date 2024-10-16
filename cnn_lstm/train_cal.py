from model import UNet_ConvLSTM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from dataset import CustomDatasetLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()


def calculate_iou(preds, labels):
    preds_int = preds.int()
    labels_int = labels.int()
    intersection = (preds_int & labels_int).float().sum((1, 2))
    union = (preds_int | labels_int).float().sum((1, 2))
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.mean()


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.0001):
    train_dataset = CustomDatasetLoader(data_path, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    val_dataset = CustomDatasetLoader(data_path, 'val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=True)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss(alpha=0.7, gamma=2.5)

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    records = {'epoch': [],
               'train_loss': [], 'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_iou': [],
               'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_iou': []}

    for epoch in range(epochs):
        net.train()
        train_metrics = {'train_loss': [], 'train_precision': [], 'train_iou': [], 'train_recall': [], 'train_f1': []}

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', total=len(train_loader)):
            images = torch.stack([image for image in images]).to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs).detach().cpu().numpy() > 5
            preds_flat = preds.reshape(-1)
            labels_np = labels.cpu().numpy()
            labels_np_flat = labels_np.reshape(-1)

            train_metrics['train_loss'].append(loss.item())
            train_metrics['train_precision'].append(precision_score(labels_np_flat, preds_flat, zero_division=1))
            train_metrics['train_iou'].append(calculate_iou(torch.tensor(preds), torch.tensor(labels_np)))
            train_metrics['train_recall'].append(recall_score(labels_np_flat, preds_flat, zero_division=1))
            train_metrics['train_f1'].append(f1_score(labels_np_flat, preds_flat))

        # Calculate and print average training metrics
        avg_train_metrics = {metric: np.mean(values) for metric, values in train_metrics.items()}
        print(f"Epoch {epoch + 1} Training - Loss: {avg_train_metrics['train_loss']:.6f}, "
              f"Precision: {avg_train_metrics['train_precision']:.6f}, "
              f"IOU: {avg_train_metrics['train_iou']:.6f}")

        net.eval()
        val_metrics = {'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_iou': []}
        with torch.no_grad():
            for images, labels in val_loader:
                images = torch.stack([image for image in images]).to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                outputs = net(images)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.4
                preds_flat = preds.reshape(-1)
                labels_np = labels.cpu().numpy()
                labels_np_flat = labels_np.reshape(-1)

                val_metrics['val_loss'].append(loss.item())
                val_metrics['val_precision'].append(precision_score(labels_np_flat, preds_flat, zero_division=1))
                val_metrics['val_recall'].append(recall_score(labels_np_flat, preds_flat, zero_division=1))
                val_metrics['val_f1'].append(f1_score(labels_np_flat, preds_flat))
                val_metrics['val_iou'].append(calculate_iou(torch.tensor(preds), torch.tensor(labels_np)))

        # Calculate and print average validation metrics
        avg_val_metrics = {metric: np.mean(values) for metric, values in val_metrics.items()}

        scheduler.step(avg_val_metrics['val_loss'])

        print(f"Epoch {epoch + 1} Validation - Loss: {avg_val_metrics['val_loss']:.6f}, "
              f"Precision: {avg_val_metrics['val_precision']:.6f}, "
              f"Recall: {avg_val_metrics['val_recall']:.6f}, F1: {avg_val_metrics['val_f1']:.6f}, "
              f"IOU: {avg_val_metrics['val_iou']:.6f}")

        if avg_val_metrics['val_precision'] > best_precision:
            best_precision = avg_val_metrics['val_precision']
            torch.save(net.state_dict(), f'best_model_p.pth')
            print(f'New best model saved with precision: {best_precision:.6f}')
        if avg_val_metrics['val_recall'] > best_recall:
            best_recall = avg_val_metrics['val_recall']
            torch.save(net.state_dict(), f'best_model_r.pth')
            print(f'New best model saved with recall: {best_recall:.6f}')
        if avg_val_metrics['val_f1'] > best_f1:
            best_f1 = avg_val_metrics['val_f1']
            torch.save(net.state_dict(), f'best_model_f.pth')
            print(f'New best model saved with f1: {best_f1:.6f}')

        # Update the records dictionary
        records['epoch'].append(epoch + 1)
        for key in avg_train_metrics:
            records[key].append(avg_train_metrics[key])
        for key in avg_val_metrics:
            records[key].append(avg_val_metrics[key])

    train_process = pd.DataFrame(records)

    return train_process


def plot_metrics(train_process):
    metrics = ['loss', 'precision', 'recall', 'f1', 'iou']
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        if train_key in train_process and val_key in train_process:
            plt.plot(train_process['epoch'].values, train_process[train_key].values, "ro-", label=f'Train {metric}')
            plt.plot(train_process['epoch'].values, train_process[val_key].values, "bs-", label=f'Val {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.title(f'Training and Validation {metric.capitalize()}')
            plt.legend()
            plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet_ConvLSTM(n_channels=3, n_classes=1, phi=2)
    net.to(device=device)
    data_path = "dataset/regular"
    # data_path = "dataset/sensitive"
    train_process = train_net(net, device, data_path, epochs=50, batch_size=4, lr=0.0001)
    plot_metrics(train_process)
