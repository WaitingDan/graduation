import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_dataset(train_dir, val_dir, batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    return train_loader, val_loader, num_classes, class_names