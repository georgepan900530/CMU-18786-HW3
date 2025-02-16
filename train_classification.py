import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from models import *
from utils import *
import argparse
import os
from tqdm import tqdm
from torchvision import models
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="SGD or Adam or AdamW",
    )
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument(
        "--model_type",
        type=str,
        default="FCNN",
        help="FCNN or CNN or ResNet18 or ResNet50 or VGG16 or EfficientNet or MyResNet",
    )
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--model_checkpoint", type=str, default=None)
    p.add_argument("--aug", action="store_true")
    p.add_argument("--save_img_path", type=str, default="./plots/plot.png")
    p.add_argument(
        "--save_model_path", type=str, default="./checkpoints/best_model.pth"
    )
    return p.parse_args()


torch.manual_seed(1124)
np.random.seed(1124)


if __name__ == "__main__":

    args = _get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.aug:
        cifar_transform_train = transforms.Compose(
            [
                transforms.RandomCrop(args.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
                ),
            ]
        )
    else:
        cifar_transform_train = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
                ),
            ]
        )

    cifar_transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    # Note that CIFAR100 contains 100 classes, each with 500 training images and 100 test images
    train_dataset = CIFAR100(
        root="./cifar_data", train=True, transform=cifar_transform_train, download=True
    )
    test_dataset = CIFAR100(
        root="./cifar_data", train=False, transform=cifar_transform_test, download=True
    )

    # Split the training dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [45000, 5000]
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize models
    if args.model_type == "FCNN":
        model = FCNN()
    elif args.model_type == "CNN":
        model = CNN(in_size=args.img_size)
    elif args.model_type == "ResNet18":
        model = ResNet18()
    elif args.model_type == "ResNet50":
        model = ResNet50()
    elif args.model_type == "VGG16":
        model = VGG16()
    elif args.model_type == "EfficientNet":
        model = EfficientNet()
    elif args.model_type == "MyResNet":
        model = MyResNet()
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    if args.model_checkpoint:
        model.load_state_dict(torch.load(args.model_checkpoint))
        print(f"Loaded model from {args.model_checkpoint}")

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=args.patience, verbose=True
    )

    logs = train_model_classification(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        args.epochs,
        args.save_model_path,
        device,
    )

    plot_loss_and_acc(logs, args.save_img_path)
