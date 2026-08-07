from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from shared.cnn import DeepChessCNN


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "cell": (ROOT / "data" / "cell_3class", 3, "cell_classifier_3class.pth"),
    "piece": (ROOT / "data" / "piece_15class", 15, "piece_classifier_15class.pth"),
    "marker": (ROOT / "data" / "marker_2class", 2, "marker_classifier_2class.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese chess cell classifier")
    parser.add_argument("kind", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir, classes, output_name = DATASETS[args.kind]
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    base_dataset = datasets.ImageFolder(data_dir)
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transform), indices[:split])
    validation_dataset = Subset(datasets.ImageFolder(data_dir, transform=validation_transform), indices[split:])
    if not validation_dataset:
        raise ValueError("Dataset needs at least two images")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DeepChessCNN(classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    loss_function = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)
    output_path = ROOT / "outputs" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    print(f"Classes: {base_dataset.classes}; device: {device}")
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                correct += int((model(images.to(device)).argmax(1).cpu() == labels).sum())
        accuracy = correct / len(validation_dataset)
        print(f"Epoch {epoch + 1}/{args.epochs}: validation accuracy {accuracy:.2%}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), output_path)
    print(f"Saved best model ({best_accuracy:.2%}): {output_path}")


if __name__ == "__main__":
    main()
