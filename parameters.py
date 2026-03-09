import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the MLP model architecture."""
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    dropout: float = 0.3
    activation: str = "relu"  # 'relu', 'gelu', 'sigmoid', 'tanh', 'leaky_relu'
    use_batchnorm: bool = True
    bn_before_activation: bool = True  # Whether to place BN before activation


@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    mode: str = "both"  # 'train', 'test', 'both'
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # Standard L2 weight decay for optimizer
    regularization_type: Optional[str] = None  # 'L1', 'L2', or None
    lambda_reg: float = 1e-5  # Coefficient for manual L1/L2 regularization
    device: str = "cpu"
    seed: int = 42
    data_dir: str = "./data"
    save_path: str = "best_model.pth"
    log_interval: int = 100
    early_stopping_patience: int = 5
    lr_scheduler_type: str = "step"  # 'constant', 'step', 'exp', 'cosine'
    lr_scheduler_gamma: float = 0.5
    lr_scheduler_step: int = 5


def get_configs():
    """Parse command line arguments and return ModelConfig and TrainingConfig objects."""
    parser = argparse.ArgumentParser(description="CS515 HW1a: MNIST MLP Classification")

    # Mode and Device
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    # Model Hyperparameters
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--activation", choices=["relu", "gelu", "sigmoid", "tanh", "leaky_relu"], default="relu")
    parser.add_argument("--no_bn", action="store_false", dest="use_batchnorm")
    parser.add_argument("--bn_after", action="store_false", dest="bn_before_activation")
    parser.set_defaults(use_batchnorm=True, bn_before_activation=True)

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--reg_type", choices=["L1", "L2", "None"], default="None")
    parser.add_argument("--lambda_reg", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--scheduler", choices=["constant", "step", "exp", "cosine"], default="step")
    parser.add_argument("--lr_step", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.5)

    args = parser.parse_args()

    model_config = ModelConfig(
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        activation=args.activation,
        use_batchnorm=args.use_batchnorm,
        bn_before_activation=args.bn_before_activation
    )

    training_config = TrainingConfig(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        regularization_type=None if args.reg_type == "None" else args.reg_type,
        lambda_reg=args.lambda_reg,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.patience,
        save_path=args.save_path,
        lr_scheduler_type=args.scheduler,
        lr_scheduler_step=args.lr_step,
        lr_scheduler_gamma=args.lr_gamma
    )

    return model_config, training_config