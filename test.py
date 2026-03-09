import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Dict

from parameters import TrainingConfig


@torch.no_grad()
def run_test(model: torch.nn.Module, config: TrainingConfig, device: torch.device):
    """
    Test the model on the MNIST test set and report accuracy.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    test_ds = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(
        test_ds, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=2
    )

    # Load the best weights
    model.load_state_dict(torch.load(config.save_path, map_location=device))
    model.eval()

    correct, n = 0, 0
    class_correct = [0] * 10
    class_total = [0] * 10

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        out = model(imgs)
        preds = out.argmax(1)
        
        correct += preds.eq(labels).sum().item()
        n += imgs.size(0)
        
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t] += 1

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(10):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  Class {i}: No samples")
            
    # Save test results for the report
    test_results = {
        'accuracy': correct / n,
        'class_accuracy': [class_correct[i] / class_total[i] for i in range(10)]
    }
    torch.save(test_results, "test_results.pth")