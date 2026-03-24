import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from experiment_utils import build_adam_step_scheduler


def get_transforms(config, train=True):
    mean, std = config.mean, config.std

    if config.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(config):
    train_tf = get_transforms(config, train=True)
    val_tf   = get_transforms(config, train=False)

    if config.dataset == "mnist":
        train_ds = datasets.MNIST(config.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(config.data_dir, train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(config.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True,  num_workers=config.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, log_interval):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


import os
import json

def save_history(history, config, name="history.json", out_dir=None):
    """Saves training history to a JSON file in the checkpoint directory."""
    target_dir = out_dir or config.checkpoint_dir
    path = os.path.join(target_dir, name)
    os.makedirs(target_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=4)

def save_history_pt(history, config, name=None, out_dir=None):
    """Saves training history to a .pt (pickle) file in checkpoint directory."""
    if name is None:
        name = f"{config.model_name}_history.pt"
    target_dir = out_dir or config.checkpoint_dir
    path = os.path.join(target_dir, name)
    os.makedirs(target_dir, exist_ok=True)
    torch.save(history, path)

def run_training(model, config, device, out_dir=None, history_prefix=None):
    if config.epochs < 1:
        raise ValueError("config.epochs must be >= 1")

    target_dir = out_dir or config.checkpoint_dir
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, config.save_path)
    
    train_loader, val_loader = get_loaders(config)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer, scheduler = build_adam_step_scheduler(model.parameters(), config)

    best_acc     = float("-inf")
    best_weights = copy.deepcopy(model.state_dict())
    history = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": [],
        "lr": []
    }
    prefix = history_prefix or config.model_name
    history_name_json = f"{prefix}_history.json"
    history_name_pt   = f"{prefix}_history.pt"

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device, config.log_interval)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        
        scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  lr: {current_lr:.6f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Incremental save
        save_history(history, config, name=history_name_json, out_dir=target_dir)
        save_history_pt(history, config, name=history_name_pt, out_dir=target_dir)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f}) to {save_path}")

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
    return {
        "history": history,
        "best_acc": float(best_acc),
        "save_path": save_path,
    }
