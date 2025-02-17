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
import matplotlib.pyplot as plt


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="FCNN",
        help="FCNN or CNN or ResNet18 or ResNet50 or VGG16 or EfficientNet or MyResNet",
    )
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--save_path", type=str, default="./plots/pred_fcnn.png")
    return p.parse_args()


if __name__ == "__main__":
    args = _get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cifar_transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    test_dataset = CIFAR100(
        root="./cifar_data", train=False, transform=cifar_transform_test, download=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

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

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    visualization_classification(model, test_loader, device, args.save_path)
