import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Optional
import numpy as np

from parameters import ModelConfig, TrainingConfig


def get_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset and return training and validation loaders.

    Args:
        config (TrainingConfig): Training configuration.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load full training set
    full_train_ds = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    
    # Split training set into train and validation 
    indices = list(range(len(full_train_ds)))
    np.random.seed(config.seed)
    np.random.shuffle(indices)
    
    val_split = 10000
    train_idx, val_idx = indices[val_split:], indices[:val_split]
    
    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_ds, val_idx)

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size,
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader


def apply_regularization(model: nn.Module, loss: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    """
    Apply manual L1 or L2 regularization to the loss.

    Args:
        model (nn.Module): The neural network model.
        loss (torch.Tensor): The current loss value.
        config (TrainingConfig): Training configuration.

    Returns:
        torch.Tensor: The loss with regularization added.
    """
    if config.regularization_type == 'L1':
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += config.lambda_reg * l1_norm
    elif config.regularization_type == 'L2':
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss += config.lambda_reg * l2_norm
    return loss


def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device, 
    config: TrainingConfig
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.
        config (TrainingConfig): Training configuration.

    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss = apply_regularization(model, loss, config)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

        if (batch_idx + 1) % config.log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    config: TrainingConfig
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.
        config (TrainingConfig): Training configuration.

    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)
            
    return total_loss / n, correct / n


def run_training(model: nn.Module, config: TrainingConfig, device: torch.device):
    """
    Run the full training process with early stopping and LR scheduling.

    Args:
        model (nn.Module): The model to train.
        config (TrainingConfig): Training configuration.
        device (torch.device): Device to run on.
    """
    train_loader, val_loader = get_loaders(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    # Scheduler selection
    if config.lr_scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.lr_scheduler_step, 
            gamma=config.lr_scheduler_gamma
        )
    elif config.lr_scheduler_type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=config.lr_scheduler_gamma
        )
    elif config.lr_scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs
        )
    else: # constant
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': None # Will be filled if mode is 'both'
    }

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        val_loss, val_acc = validate(model, val_loader, criterion, device, config)
        
        if config.lr_scheduler_type != 'constant':
            scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Early Stopping and Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, config.save_path)
            print(f"  --> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # Save history for visualization
    torch.save(history, "train_history.pth")
    print(f"\nTraining done. Best validation loss: {best_val_loss:.4f}")