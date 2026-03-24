import argparse
import torch
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ExperimentConfig:
    # Data params
    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 2
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010)
    
    # Model params
    model_name: str = "resnet"
    num_classes: int = 10
    input_size: int = 3072
    vgg_depth: str = "16"
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    
    # Training params
    mode: str = "both"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    
    # KD params
    is_kd: bool = False
    temperature: float = 3.0
    alpha: float = 0.5
    teacher_path: str = ""
    
    # Misc
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "best_model.pth"
    checkpoint_dir: str = ".checkpoints"
    log_interval: int = 100

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_params() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Deep Learning Research Assignment (HW1b)")

    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"], default="resnet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--scheduler_step_size", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    
    # VGG/ResNet specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2])

    args = parser.parse_args()

    config = ExperimentConfig(
        mode=args.mode,
        dataset=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers
    )

    if args.device:
        config.device = args.device

    if config.dataset == "mnist":
        config.input_size = 784
        config.mean = (0.1307,)
        config.std = (0.3081,)
    
    set_seed(config.seed)
    return config
