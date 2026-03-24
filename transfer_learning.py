import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from parameters import set_seed
from train import train_one_epoch, validate, save_history
from experiment_utils import build_adam_step_scheduler
import time
import os

def get_resnet18_option1(num_classes=10):
    """
    Option 1: Load pretrained ResNet-18, freeze all layers except the FC layer.
    Expects 224x224 input (resized from 32x32).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_resnet18_option2(num_classes=10):
    """
    Option 2: Load pretrained ResNet-18, modify the first conv layer for 32x32 input, 
    and fine-tune the entire network.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Original ResNet-18 starts with 7x7 conv, stride 2, padding 3 for 224x224 input.
    # For 32x32, we replace it with 3x3 conv, stride 1, padding 1 to preserve spatial resolution.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() # Remove maxpool which would reduce 32x32 too aggressively early on
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Fine-tune everything
    for param in model.parameters():
        param.requires_grad = True
        
    return model

def run_experiment(option_name, model_fn, transform, config, device, out_dir=None):
    if config.epochs < 1:
        raise ValueError("config.epochs must be >= 1")

    print(f"\n--- Running Experiment: {option_name} ---")
    target_dir = out_dir or config.checkpoint_dir
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, f"resnet18_{option_name}_best.pth")
    
    set_seed(config.seed)
    model = model_fn(config.num_classes).to(device)
    
    train_ds = torchvision.datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=transform['train'])
    val_ds = torchvision.datasets.CIFAR10(root=config.data_dir, train=False, download=True, transform=transform['val'])
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    optimizer, scheduler = build_adam_step_scheduler(filter(lambda p: p.requires_grad, model.parameters()), config)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = float("-inf")
    start_time = time.time()
    history = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": [],
        "lr": []
    }
    history_name = f"resnet18_{option_name}_history.json"
    
    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, config.log_interval)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Val Acc {val_acc:.4f}, LR {current_lr:.6f}")
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        # Incremental save
        save_history(history, config, name=history_name, out_dir=target_dir)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f}) to {save_path}")

        scheduler.step()
            
    total_time = time.time() - start_time
    print(f"Experiment {option_name} finished. Best Acc: {best_acc:.4f}, Time: {total_time:.2f}s")
    return {
        "best_acc": float(best_acc),
        "time_sec": float(total_time),
        "history": history,
        "save_path": save_path,
        "history_path": os.path.join(target_dir, history_name),
    }

if __name__ == "__main__":
    raise RuntimeError(
        "Direct execution is disabled. Use `python experiment.py run-suite --suite hw1b` "
        "or `python experiment.py run-single ...`."
    )
