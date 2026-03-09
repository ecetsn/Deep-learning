import random
import ssl
import numpy as np
import torch

from parameters import get_configs
from mlp_model import MLP
from train import run_training
from test import run_test

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main entry point for the MNIST MLP classification."""
    model_config, training_config = get_configs()

    set_seed(training_config.seed)
    print(f"Seed set to: {training_config.seed}")
    
    device = torch.device(
        training_config.device if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    print(f"Using device: {device}")

    # Build model
    model = MLP(model_config).to(device)
    print(model)

    # Counting parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    if training_config.mode in ("train", "both"):
        print("\nStarting Training...")
        run_training(model, training_config, device)

    if training_config.mode in ("test", "both"):
        print("\nStarting Testing...")
        run_test(model, training_config, device)


if __name__ == "__main__":
    main()