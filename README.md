# CS515 Deep Learning - Assignment 1: MNIST MLP Analysis

This repository contains the implementation and analysis of a Multi-Layer Perceptron (MLP) for the MNIST digit classification task, as part of the CS515 Deep Learning course. The project includes a flexible MLP architecture, an automated experiment runner for ablation studies, and a comprehensive analysis notebook.

## Repository Structure

```text
.
├── data/                       # MNIST dataset (auto-downloaded)
├── experiments/                # Storage for trained models and results
│   ├── models/                 # Saved .pth model files
│   └── results/                # Phase-wise training history and test results
├── images/                     # Visualization images for documentation
├── mlp_model.py                # Core MLP architecture implementation
├── main.py                     # Entry point for training and testing
├── parameters.py               # Hyperparameter configurations and CLI parsing
├── run_experiments.py          # Script to automate multiple experimental phases
├── train.py                    # Training and validation logic
├── test.py                     # Testing and metric calculation logic
├── mlp_model_analysis.ipynb    # Detailed visualization and analysis notebook
└── requirements.txt            # Project dependencies
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd Deep-learning
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/bin/activate  # On Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Default Model Parameters

The default configuration (defined in `parameters.py`) used as the baseline:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `hidden_sizes` | `[512, 256, 128]` | Neurons per hidden layer |
| `activation` | `relu` | Hidden layer activation function |
| `dropout` | `0.3` | Dropout probability |
| `use_batchnorm`| `True` | Apply Batch Normalization |
| `epochs` | `20` | Max training epochs |
| `batch_size` | `64` | Mini-batch size |
| `learning_rate`| `1e-3` | Initial Adam learning rate |
| `weight_decay` | `1e-4` | L2 weight decay (optimizer) |

## Usage

### Single Run
To train and test a model with default or custom parameters:
```bash
python main.py --mode both --epochs 10 --activation gelu
```

### Running Experiments (Ablation Studies)
The `run_experiments.py` script automates the process of testing various configurations across different phases:
- **Phase 1**: Architecture Search (Depth vs. Width)
- **Phase 2**: Activations & Batch Normalization placement
- **Phase 3**: Dropout & Manual Regularization (L1/L2)
- **Phase 4**: Training Framework (Learning Rates, Schedulers, Patience)

Run the full suite with:
```bash
python run_experiments.py
```
Results will be saved in `experiments/results/`.

## Model Analysis & Visualization

Detailed analysis is provided in the [mlp_model_analysis.ipynb](mlp_model_analysis.ipynb) notebook. It covers:

1.  **Ablation Study Results**: Comparative charts for all experimental phases.
2.  **Training Dynamics**: Training/Validation loss and accuracy curves.
3.  **Representation Learning**: **t-SNE visualization** showing how the MLP clusters different digits in the hidden space.
4.  **Error Analysis**: Confusion matrices and examples of misclassified digits.
5.  **Statistical Insights**: Confidence intervals and weight distribution analysis.