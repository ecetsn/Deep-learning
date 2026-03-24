import ssl
import torch
from parameters import get_params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from train import run_training
from evaluate import run_test

# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context

def build_model(config):
    model_name = config.model_name
    dataset    = config.dataset
    nc         = config.num_classes

    if model_name == "mlp":
        return MLP(
            input_size   = config.input_size,
            hidden_sizes = [512, 256, 128], # Default hidden sizes as per old parameters.py
            num_classes  = nc,
            dropout      = 0.3,             # Default dropout as per old parameters.py
        )

    if model_name == "cnn":
        # MNIST_CNN expects 1-channel 28×28; SimpleCNN expects 3-channel 32×32
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        else:
            return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError("VGG is designed for 3-channel images; use cifar10 with vgg.")
        return VGG(dept=config.vgg_depth, num_class=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError("ResNet is designed for 3-channel images; use cifar10 with resnet.")
        return ResNet(BasicBlock, config.resnet_layers, num_classes=nc)
        
    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError("MobileNetV2 is designed for 3-channel images; use cifar10 with mobilenet.")
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")

def main():
    config = get_params()

    print(f"Seed set to: {config.seed}")
    print(f"Dataset: {config.dataset}  |  Model: {config.model_name}")
    print(f"Using device: {config.device}")

    device = torch.device(config.device)
    model = build_model(config).to(device)
    
    # Optional: Print model summary using ptflops if in train mode
    if config.mode in ("train", "both"):
        try:
            from ptflops import get_model_complexity_info
            input_res = (1, 28, 28) if config.dataset == "mnist" else (3, 32, 32)
            macs, params = get_model_complexity_info(model, input_res, as_strings=True, print_per_layer_stat=False)
            print(f"Computational complexity: {macs}")
            print(f"Number of parameters: {params}")
        except Exception as e:
            print(f"Could not calculate complexity: {e}")

    if config.mode in ("train", "both"):
        run_training(model, config, device)

    if config.mode in ("test", "both"):
        run_test(model, config, device)

if __name__ == "__main__":
    main()