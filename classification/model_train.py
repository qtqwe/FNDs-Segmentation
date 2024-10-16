import copy
import time

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
from ViT_model import CustomViT
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm


def train_val_data_process():
    # Define the path for the training dataset
    ROOT_TRAIN = 'classify_data/train'
    ROOT_VAL = 'classify_data/val'

    # Define normalization operations
    normalize = transforms.Normalize([0.0717982,  0.00635558, 0.01404482], [0.02460866, 0.00014787, 0.00068004])
    # Define data preprocessing operations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    # Load training data
    train_data = ImageFolder(ROOT_TRAIN, transform=data_transform)
    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=12, shuffle=True, num_workers=10)
    # Split data into training and validation sets
    val_data = ImageFolder(ROOT_VAL, transform=data_transform)
    # Create DataLoader
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size=12, shuffle=False, num_workers=10)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Move the model to the specified device
    model = model.to(device)
    # Save the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()

    # Prepare to track the predictions
    all_preds = []
    all_targets = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{num_epochs - 1}', unit='batch') as pbar:
            model.train()
            for step, (b_x, b_y) in enumerate(train_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)

                pbar.update(1)
                pbar.set_postfix(
                    {'train_loss': train_loss / train_num, 'train_acc': train_corrects.double().item() / train_num})

            model.eval()
            with torch.no_grad():
                for step, (b_x, b_y) in enumerate(val_dataloader):
                    b_x = b_x.to(device)
                    b_y = b_y.to(device)

                    output = model(b_x)
                    pre_lab = torch.argmax(output, dim=1)
                    loss = criterion(output, b_y)

                    val_loss += loss.item() * b_x.size(0)
                    val_corrects += torch.sum(pre_lab == b_y.data)
                    val_num += b_x.size(0)

                    # Track the predictions and their true labels
                    all_preds.extend(pre_lab.view(-1).cpu().numpy())
                    all_targets.extend(b_y.cpu().numpy())

                pbar.update(1)
                pbar.set_postfix({'val_loss': val_loss / val_num, 'val_acc': val_corrects.double().item() / val_num})

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f"{epoch} train loss: {train_loss_all[-1]:.4f} train acc: {train_acc_all[-1]:.4f}")
        print(f"{epoch} val loss: {val_loss_all[-1]:.4f} val acc: {val_acc_all[-1]:.4f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    time_use = time.time() - since
    print(f"Training and validation took {time_use // 60:.0f}m {time_use % 60:.0f}s")

    torch.save(best_model_wts, "best_model_ViT_8.pth")

    # Generate and plot the confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(1, 8), yticklabels=range(1, 8))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all, })

    return train_process


def matplot_acc_loss(train_process):
    # Display training and validation loss and accuracy after each iteration
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'].values, train_process['train_loss_all'].values, "ro-", label="Train loss")
    plt.plot(train_process['epoch'].values, train_process['val_loss_all'].values, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'].values, train_process['train_acc_all'].values, "ro-", label="Train acc")
    plt.plot(train_process['epoch'].values, train_process['val_acc_all'].values, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load the required model
    ViT = CustomViT(8)
    # Load dataset
    train_data, val_data = train_val_data_process()
    # Train the model using the existing setup
    train_process = train_model_process(ViT, train_data, val_data, num_epochs=50)
    matplot_acc_loss(train_process)
