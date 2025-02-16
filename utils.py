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
    max_test_acc = float("-inf")
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

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            print(f"Saving model at epoch {epoch+1}")
            torch.save(model.state_dict(), save_model_path)

        scheduler.step(test_acc)

        tqdm.write(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%"
        )
    return logs


def visualization_classification(model, test_loader, device, save_path):
    """
    In this function, we will need to visualize a single image from each class in the test set.
    Then, show the ground truth label and the prediction of the model for each image.
    """
    cifar100_classes = {
        0: "apple",
        1: "aquarium_fish",
        2: "baby",
        3: "bear",
        4: "beaver",
        5: "bed",
        6: "bee",
        7: "beetle",
        8: "bicycle",
        9: "bottle",
        10: "bowl",
        11: "boy",
        12: "bridge",
        13: "bus",
        14: "butterfly",
        15: "camel",
        16: "can",
        17: "castle",
        18: "caterpillar",
        19: "cattle",
        20: "chair",
        21: "chimpanzee",
        22: "clock",
        23: "cloud",
        24: "cockroach",
        25: "couch",
        26: "crab",
        27: "crocodile",
        28: "cup",
        29: "dinosaur",
        30: "dolphin",
        31: "elephant",
        32: "flatfish",
        33: "forest",
        34: "fox",
        35: "girl",
        36: "hamster",
        37: "house",
        38: "kangaroo",
        39: "keyboard",
        40: "lamp",
        41: "lawn_mower",
        42: "leopard",
        43: "lion",
        44: "lizard",
        45: "lobster",
        46: "man",
        47: "maple_tree",
        48: "motorcycle",
        49: "mountain",
        50: "mouse",
        51: "mushroom",
        52: "oak_tree",
        53: "orange",
        54: "orchid",
        55: "otter",
        56: "palm_tree",
        57: "pear",
        58: "pickup_truck",
        59: "pine_tree",
        60: "plain",
        61: "plate",
        62: "poppy",
        63: "porcupine",
        64: "possum",
        65: "rabbit",
        66: "raccoon",
        67: "ray",
        68: "road",
        69: "rocket",
        70: "rose",
        71: "sea",
        72: "seal",
        73: "shark",
        74: "shrew",
        75: "skunk",
        76: "skyscraper",
        77: "snail",
        78: "snake",
        79: "spider",
        80: "squirrel",
        81: "streetcar",
        82: "sunflower",
        83: "sweet_pepper",
        84: "table",
        85: "tank",
        86: "telephone",
        87: "television",
        88: "tiger",
        89: "tractor",
        90: "train",
        91: "trout",
        92: "tulip",
        93: "turtle",
        94: "wardrobe",
        95: "whale",
        96: "willow_tree",
        97: "wolf",
        98: "woman",
        99: "worm",
    }
    model.eval()
    predictions = {}
    labels = {}
    class_images = {}
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        outputs = model(img)
        pred = torch.argmax(outputs, dim=1)
        if label.item() not in class_images:
            class_images[label.item()] = img[0].cpu().numpy()
            labels[label.item()] = label.item()
            predictions[label.item()] = pred.item()

        if len(class_images) == 100:
            break

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, (label, image) in enumerate(class_images.items()):
        ax = axes[i // 10, i % 10]
        image = image.transpose(1, 2, 0)  # CxHxW -> HxWxC
        image = image * np.array([0.2675, 0.2565, 0.2761]) + np.array(
            [0.5071, 0.4867, 0.4408]
        )  # Unnormalize
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.set_title(
            f"Class: {cifar100_classes[label]}\nGT: {labels[label]}\nPred: {predictions[label]}",
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
