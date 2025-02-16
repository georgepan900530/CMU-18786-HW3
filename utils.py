import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def plot_loss_and_acc(logs, save_path=None):
    """
    Function to plot training and validation/test loss curves
    :param logs: dict with keys 'train_loss','test_loss' and 'epochs', where train_loss and test_loss are lists with
                            the training and test/validation loss for each epoch
    """
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    t = np.arange(len(logs["train_loss"]))
    plt.plot(t, logs["train_loss"], label="train_loss", lw=3)
    plt.plot(t, logs["val_loss"], label="val_loss", lw=3)
    plt.plot(t, logs["test_loss"], label="test_loss", lw=3)
    plt.grid(1)
    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("loss value", fontsize=15)
    plt.legend(fontsize=15)

    plt.subplot(1, 2, 2)
    t = np.arange(len(logs["train_acc"]))
    plt.plot(t, logs["train_acc"], label="train_acc", lw=3)
    plt.plot(t, logs["val_acc"], label="val_acc", lw=3)
    plt.plot(t, logs["test_acc"], label="test_acc", lw=3)
    plt.grid(1)
    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("accuracy", fontsize=15)
    plt.legend(fontsize=15)

    plt.savefig(save_path)


def train_model_classification(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    save_model_path,
    device,
):
    model.train()
    logs = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epochs": [i + 1 for i in range(num_epochs)],
    }
    min_val_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        train_loss_list = []
        train_acc_list = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)

            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).sum().item() / imgs.shape[0]
            train_acc_list.append(acc)

        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = sum(train_acc_list) / len(train_acc_list)
        logs["train_loss"].append(train_loss)
        logs["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss_list = []
        val_acc_list = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss_list.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).sum().item() / imgs.shape[0]
                val_acc_list.append(acc)

        val_loss = sum(val_loss_list) / len(val_loss_list)
        val_acc = sum(val_acc_list) / len(val_acc_list)
        logs["val_loss"].append(val_loss)
        logs["val_acc"].append(val_acc)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print(f"Saving model at epoch {epoch+1}")
            torch.save(model.state_dict(), save_model_path)

        scheduler.step(val_loss)

        # Testing
        model.eval()
        test_loss_list = []
        test_acc_list = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                test_loss_list.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == labels).sum().item() / imgs.shape[0]
                test_acc_list.append(acc)
        test_loss = sum(test_loss_list) / len(test_loss_list)
        test_acc = sum(test_acc_list) / len(test_acc_list)
        logs["test_loss"].append(test_loss)
        logs["test_acc"].append(test_acc)

        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%"
        )
    return logs


def visualization_classification(model, test_loader, device):
    pass
