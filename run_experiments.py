import subprocess
import os
import json

def run_experiment(phase, name, args):
    print(f"\n>>> Phase: {phase} | Experiment: {name}")
    
    # Define paths for this experiment
    results_dir = f"experiments/results/{phase}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)
    
    model_path = f"experiments/models/{name}.pth"
    history_path = f"{results_dir}/{name}_history.pth"
    
    # Add save_path and device to args
    cmd_args = args + ["--save_path", model_path, "--mode", "both", "--device", "cuda"]
    cmd = ["venv/bin/python", "main.py"] + cmd_args # use main.py for each experiment
    subprocess.run(cmd)
    
    # Move history and test results to the correct location
    if os.path.exists("train_history.pth"):
        os.rename("train_history.pth", history_path)
    if os.path.exists("test_results.pth"):
        os.rename("test_results.pth", f"{results_dir}/{name}_test.pth")

# Architecture Search
arch_base = ["--activation", "relu", "--no_bn", "--dropout", "0", "--epochs", "10"]
depths = [1, 2, 3, 4, 5, 6]
widths = [128, 256, 512, 1024]

for d in depths:
    for w in widths:
        name = f"depth{d}_width{w}"
        hidden = [str(w)] * d
        run_experiment("phase1", name, arch_base + ["--hidden_sizes"] + hidden)

# best architecture from dry run of the previous step
best_arch = ["--hidden_sizes", "512", "512", "512", "--epochs", "10"] 

# Activation & BN
activations = ["relu", "gelu", "sigmoid", "tanh", "leaky_relu"]
bn_configs = [
    ("no_bn", ["--no_bn"]),
    ("bn_before", []), # default is BN before
    ("bn_after", ["--bn_after"])
]

for act in activations:
    run_experiment("phase2_act", act, best_arch + ["--activation", act, "--no_bn"])

for bn_name, bn_args in bn_configs:
    run_experiment("phase2_bn", bn_name, best_arch + ["--activation", "relu"] + bn_args)

# Dropout & Regularization
dropout_rates = [0, 0.2, 0.4, 0.6, 0.8]
for dr in dropout_rates:
    run_experiment("phase3_dropout", f"dropout_{dr}", best_arch + ["--dropout", str(dr)])

regs = ["L1", "L2"]
lambdas = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1"]
for rt in regs:
    for lb in lambdas:
        run_experiment("phase3_reg", f"{rt}_{lb}", best_arch + ["--reg_type", rt, "--lambda_reg", lb])

# Trainning Framework
lrs = ["1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
for lr in lrs:
    run_experiment("phase4_lr", f"lr_{lr}", best_arch + ["--lr", lr])

schedulers = ["constant", "step", "exp", "cosine"]
for scheduler in schedulers:
    run_experiment("phase4_scheduler", f"sched_{scheduler}", best_arch + ["--scheduler", scheduler])

patience_values = [5, 10, 20]
for p in patience_values:
    run_experiment("phase4_patience", f"patience_{p}", best_arch + ["--patience", str(p)])
