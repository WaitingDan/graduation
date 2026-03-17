import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.plot_results import plot_curve
from utils.common import get_device, build_default_transforms, write_class_indices

from models.resnet_model import create_resnet


def main():

    device = get_device()
    print("Using device:", device)

    train_dir = os.path.join(ROOT_DIR, "dataset/ship_42/train")
    val_dir = os.path.join(ROOT_DIR, "dataset/ship_42/val")

    transform = build_default_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform["train"])
    val_dataset = datasets.ImageFolder(val_dir, transform["val"])

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    write_class_indices(class_names, os.path.join(ROOT_DIR, "class_indices.json"))

    model = create_resnet(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 15
    best_acc = 0

    weight_path = os.path.join(ROOT_DIR, "weights/resnet_best.pth")
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        model.train()

        train_bar = tqdm(train_loader, file=sys.stdout)

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()

        val_bar = tqdm(val_loader, file=sys.stdout)

        val_loss_sum = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_bar:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_sum += loss.item()

                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / len(val_loader)
        val_acc = correct / total

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")

        if val_acc > best_acc:

            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)

            print("Saved Best Model")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    plot_curve(train_loss_list, val_loss_list, "loss", "resnet")
    plot_curve(train_acc_list, val_acc_list, "accuracy", "resnet")


if __name__ == "__main__":
    main()