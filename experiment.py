import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from evaluate import evaluate_model
from experiment_utils import (
    best_epoch_and_acc,
    get_transfer_transforms,
    canonical_run_id,
    ensure_dir,
    model_complexity,
    now_timestamp,
    rows_to_csv,
    save_json_atomic,
    save_numpy,
    save_npz,
)
from knowledge_distillation import run_kd_experiment
from main import build_model
from models.CNN import SimpleCNN
from models.ResNet import BasicBlock, ResNet
from models.mobilenet import MobileNetV2
from parameters import ExperimentConfig, set_seed
from train import run_training
from transfer_learning import get_resnet18_option1, get_resnet18_option2, run_experiment


def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    if probs.size == 0 or labels.size == 0:
        return {"ece": 0.0, "brier": 0.0, "nll": 0.0}

    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (conf >= left) & (conf < right if i < n_bins - 1 else conf <= right)
        if not np.any(mask):
            continue
        bin_acc = acc[mask].mean()
        bin_conf = conf[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * (mask.sum() / len(conf))

    one_hot = np.eye(probs.shape[1], dtype=np.float32)[labels]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    true_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    nll = float(-np.mean(np.log(true_probs)))
    return {"ece": float(ece), "brier": brier, "nll": nll}


def make_base_cfg(seed: int, epochs: int, lr: float = 1e-3, label_smoothing: float = 0.0) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.dataset = "cifar10"
    cfg.seed = seed
    cfg.epochs = epochs
    cfg.learning_rate = lr
    cfg.label_smoothing = label_smoothing
    cfg.mode = "train"
    cfg.scheduler_step_size = 5
    cfg.scheduler_gamma = 0.5
    cfg.save_path = "checkpoint_best.pth"
    return cfg


def _copy_checkpoint(src: str, dst: str) -> None:
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    shutil.copy2(src, dst)


def write_run_artifacts(
    run_dir: str,
    cfg: ExperimentConfig,
    history: Dict[str, Any],
    timing: Dict[str, Any],
    complexity: Dict[str, Any],
    eval_result: Dict[str, Any],
    checkpoint_src: str,
) -> Dict[str, Any]:
    ensure_dir(run_dir)
    config_path = os.path.join(run_dir, "config.json")
    history_path = os.path.join(run_dir, "history.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    complexity_path = os.path.join(run_dir, "complexity.json")
    timing_path = os.path.join(run_dir, "timing.json")
    pred_path = os.path.join(run_dir, "predictions_test.npz")
    cm_path = os.path.join(run_dir, "confusion_matrix_test.npy")
    feat_path = os.path.join(run_dir, "features_test.npz")
    ckpt_dst = os.path.join(run_dir, "checkpoint_best.pth")

    _copy_checkpoint(checkpoint_src, ckpt_dst)

    run_cfg = asdict(cfg)
    save_json_atomic(config_path, run_cfg)
    save_json_atomic(history_path, history)
    save_json_atomic(complexity_path, complexity)
    save_json_atomic(timing_path, timing)

    labels = eval_result["predictions"]["labels"]
    preds = eval_result["predictions"]["preds"]
    logits = eval_result["predictions"]["logits"]
    probs = eval_result["predictions"]["probs"]

    save_npz(pred_path, labels=labels, preds=preds, logits=logits, probs=probs)
    cm = confusion_matrix(labels, preds)
    save_numpy(cm_path, cm)

    features = eval_result.get("features", {})
    if "features" in features and features["features"].size > 0:
        save_npz(feat_path, features=features["features"], labels=features["labels"])

    best_info = best_epoch_and_acc(history)
    cal = compute_calibration_metrics(probs=probs, labels=labels)
    metrics = {
        "overall_accuracy_test": eval_result["overall_accuracy"],
        "class_accuracy_test": eval_result["class_accuracy"],
        "class_correct_test": eval_result["class_correct"],
        "class_total_test": eval_result["class_total"],
        "best_epoch": best_info["best_epoch"],
        "best_val_acc": best_info["best_val_acc"],
        "final_val_acc": best_info["final_val_acc"],
        "final_train_acc": float(history["train_acc"][-1]) if history.get("train_acc") else None,
        "ece": cal["ece"],
        "brier": cal["brier"],
        "nll": cal["nll"],
    }
    save_json_atomic(metrics_path, metrics)
    return metrics


def run_baseline(run_dir: str, experiment_name: str, seed: int, model_name: str, epochs: int, label_smoothing: float) -> Dict[str, Any]:
    cfg = make_base_cfg(seed=seed, epochs=epochs, label_smoothing=label_smoothing)
    cfg.model_name = model_name
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    start = time.time()
    out = run_training(model, cfg, device, out_dir=run_dir, history_prefix=experiment_name)
    total_sec = time.time() - start
    eval_result = evaluate_model(
        model,
        cfg,
        device,
        checkpoint_path=out["save_path"],
        collect_predictions=True,
        collect_features=True,
    )
    complexity = model_complexity(model)
    timing = {
        "train_wall_time_sec": float(total_sec),
        "epochs": int(cfg.epochs),
        "avg_epoch_time_sec": float(total_sec / cfg.epochs if cfg.epochs else 0.0),
    }
    metrics = write_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        history=out["history"],
        timing=timing,
        complexity=complexity,
        eval_result=eval_result,
        checkpoint_src=out["save_path"],
    )
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "run_dir": run_dir,
        "metrics": metrics,
        "complexity": complexity,
        "timing": timing,
        "checkpoint": os.path.join(run_dir, "checkpoint_best.pth"),
    }


def run_transfer(run_dir: str, experiment_name: str, seed: int, option: str, epochs: int) -> Dict[str, Any]:
    cfg = make_base_cfg(seed=seed, epochs=epochs)
    cfg.model_name = "resnet"
    cfg.dataset = "cifar10"
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    tf = get_transfer_transforms()

    if option == "option1":
        out = run_experiment("Option1_Resize", get_resnet18_option1, tf["option1"], cfg, device, out_dir=run_dir)
        model = get_resnet18_option1(cfg.num_classes).to(device)
        eval_tf = tf["option1"]["val"]
    elif option == "option2":
        out = run_experiment("Option2_Modify", get_resnet18_option2, tf["option2"], cfg, device, out_dir=run_dir)
        model = get_resnet18_option2(cfg.num_classes).to(device)
        eval_tf = tf["option2"]["val"]
    else:
        raise ValueError(f"Unknown transfer option: {option}")

    checkpoint = out["save_path"]
    history = out["history"]
    eval_result = evaluate_model(
        model,
        cfg,
        device,
        checkpoint_path=checkpoint,
        collect_predictions=True,
        collect_features=True,
        custom_transform=eval_tf,
    )
    complexity = model_complexity(model)
    timing = {
        "train_wall_time_sec": float(out["time_sec"]),
        "epochs": int(cfg.epochs),
        "avg_epoch_time_sec": float(out["time_sec"] / cfg.epochs if cfg.epochs else 0.0),
    }
    metrics = write_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        history=history,
        timing=timing,
        complexity=complexity,
        eval_result=eval_result,
        checkpoint_src=checkpoint,
    )
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "run_dir": run_dir,
        "metrics": metrics,
        "complexity": complexity,
        "timing": timing,
        "checkpoint": os.path.join(run_dir, "checkpoint_best.pth"),
    }


def _load_teacher(teacher_checkpoint: str, device: torch.device) -> torch.nn.Module:
    if not os.path.exists(teacher_checkpoint):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_checkpoint}. "
            "KD requires a trained ResNet teacher checkpoint."
        )
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
    teacher.eval()
    return teacher


def run_kd(
    run_dir: str,
    experiment_name: str,
    seed: int,
    custom: bool,
    epochs: int,
    teacher_checkpoint: str,
) -> Dict[str, Any]:
    cfg = make_base_cfg(seed=seed, epochs=epochs)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    teacher = _load_teacher(teacher_checkpoint=teacher_checkpoint, device=device)
    student = MobileNetV2(num_classes=10) if custom else SimpleCNN(num_classes=10)
    start = time.time()
    out = run_kd_experiment(
        name="Custom_KD_MobileNet" if custom else "Standard_KD_CNN",
        student=student,
        teacher=teacher,
        config=cfg,
        custom=custom,
        out_dir=run_dir,
    )
    total_sec = time.time() - start
    eval_result = evaluate_model(
        student,
        cfg,
        device,
        checkpoint_path=out["save_path"],
        collect_predictions=True,
        collect_features=True,
    )
    complexity = model_complexity(student)
    timing = {
        "train_wall_time_sec": float(total_sec),
        "epochs": int(cfg.epochs),
        "avg_epoch_time_sec": float(total_sec / cfg.epochs if cfg.epochs else 0.0),
    }
    metrics = write_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        history=out["history"],
        timing=timing,
        complexity=complexity,
        eval_result=eval_result,
        checkpoint_src=out["save_path"],
    )
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "run_dir": run_dir,
        "metrics": metrics,
        "complexity": complexity,
        "timing": timing,
        "checkpoint": os.path.join(run_dir, "checkpoint_best.pth"),
    }


def suite_matrix_hw1b() -> List[Tuple[str, int]]:
    rows = []
    rows.append(("resnet18_option1_resize", 42))
    rows.append(("resnet18_option2_modify", 42))
    for s in (42, 43, 44):
        rows.append(("cnn_baseline", s))
        rows.append(("resnet_no_smoothing", s))
        rows.append(("resnet_label_smoothing", s))
        rows.append(("kd_standard_cnn_student", s))
        rows.append(("kd_custom_mobilenet_student", s))
    return rows


def run_single(experiment_root: str, experiment_name: str, seed: int) -> Dict[str, Any]:
    run_id = canonical_run_id(experiment_name, seed)
    run_dir = os.path.join(experiment_root, run_id)
    ensure_dir(run_dir)

    if experiment_name == "cnn_baseline":
        return run_baseline(run_dir, experiment_name, seed, model_name="cnn", epochs=20, label_smoothing=0.0)
    if experiment_name == "resnet_no_smoothing":
        return run_baseline(run_dir, experiment_name, seed, model_name="resnet", epochs=20, label_smoothing=0.0)
    if experiment_name == "resnet_label_smoothing":
        return run_baseline(run_dir, experiment_name, seed, model_name="resnet", epochs=20, label_smoothing=0.1)
    if experiment_name == "resnet18_option1_resize":
        return run_transfer(run_dir, experiment_name, seed, option="option1", epochs=5)
    if experiment_name == "resnet18_option2_modify":
        return run_transfer(run_dir, experiment_name, seed, option="option2", epochs=5)
    if experiment_name in ("kd_standard_cnn_student", "kd_custom_mobilenet_student"):
        teacher_run = canonical_run_id("resnet_no_smoothing", seed)
        teacher_ckpt = os.path.join(experiment_root, teacher_run, "checkpoint_best.pth")
        return run_kd(
            run_dir=run_dir,
            experiment_name=experiment_name,
            seed=seed,
            custom=(experiment_name == "kd_custom_mobilenet_student"),
            epochs=20,
            teacher_checkpoint=teacher_ckpt,
        )
    raise ValueError(f"Unknown experiment: {experiment_name}")


def _collect_summary_rows(experiment_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_id in sorted(os.listdir(experiment_root)):
        run_dir = os.path.join(experiment_root, run_id)
        if not os.path.isdir(run_dir):
            continue
        req = ["config.json", "history.json", "metrics.json", "complexity.json", "timing.json"]
        if not all(os.path.exists(os.path.join(run_dir, f)) for f in req):
            continue
        with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(os.path.join(run_dir, "complexity.json"), "r", encoding="utf-8") as f:
            complexity = json.load(f)
        with open(os.path.join(run_dir, "timing.json"), "r", encoding="utf-8") as f:
            timing = json.load(f)

        experiment_name = run_id.split("__seed")[0]
        seed = int(run_id.split("__seed")[1]) if "__seed" in run_id else cfg.get("seed", -1)
        row = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "seed": seed,
            "best_epoch": metrics.get("best_epoch"),
            "best_val_acc": metrics.get("best_val_acc"),
            "final_val_acc": metrics.get("final_val_acc"),
            "overall_accuracy_test": metrics.get("overall_accuracy_test"),
            "ece": metrics.get("ece"),
            "brier": metrics.get("brier"),
            "nll": metrics.get("nll"),
            "mmacs": complexity.get("mmacs"),
            "mparams": complexity.get("mparams"),
            "train_wall_time_sec": timing.get("train_wall_time_sec"),
        }
        rows.append(row)
    return rows


def summarize(experiment_root: str) -> None:
    rows = _collect_summary_rows(experiment_root)
    rows = sorted(rows, key=lambda r: r["run_id"])
    summary_json = os.path.join(experiment_root, "summary.json")
    summary_csv = os.path.join(experiment_root, "summary.csv")
    save_json_atomic(summary_json, {"rows": rows})
    cols = [
        "run_id",
        "experiment_name",
        "seed",
        "best_epoch",
        "best_val_acc",
        "final_val_acc",
        "overall_accuracy_test",
        "ece",
        "brier",
        "nll",
        "mmacs",
        "mparams",
        "train_wall_time_sec",
    ]
    rows_to_csv(summary_csv, rows, cols)


def run_suite(suite: str, output_root: str) -> str:
    if suite != "hw1b":
        raise ValueError("Only --suite hw1b is supported.")
    ts = now_timestamp()
    experiment_root = os.path.join(output_root, f"{ts}_hw1b_suite")
    ensure_dir(experiment_root)

    executed = []
    for experiment_name, seed in suite_matrix_hw1b():
        info = run_single(experiment_root, experiment_name, seed)
        executed.append(
            {
                "run_id": canonical_run_id(experiment_name, seed),
                "experiment_name": experiment_name,
                "seed": seed,
                "run_dir": info["run_dir"],
            }
        )

    save_json_atomic(
        os.path.join(experiment_root, "suite_manifest.json"),
        {"suite": suite, "experiment_root": experiment_root, "runs": executed},
    )
    summarize(experiment_root)
    return experiment_root


def main():
    parser = argparse.ArgumentParser(description="HW1b structured experiment runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_suite = sub.add_parser("run-suite", help="Run preset experiment suite")
    p_suite.add_argument("--suite", type=str, default="hw1b")
    p_suite.add_argument("--output-root", type=str, default="experiments")

    p_single = sub.add_parser("run-single", help="Run a single experiment")
    p_single.add_argument("--experiment", type=str, required=True)
    p_single.add_argument("--seed", type=int, required=True)
    p_single.add_argument("--experiment-root", type=str, default=None)
    p_single.add_argument("--output-root", type=str, default="experiments")

    p_sum = sub.add_parser("summarize", help="Build summary files from an experiment root")
    p_sum.add_argument("--experiment-root", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "run-suite":
        root = run_suite(suite=args.suite, output_root=args.output_root)
        print(f"Suite completed: {root}")
        return

    if args.cmd == "run-single":
        if args.experiment_root:
            root = args.experiment_root
            ensure_dir(root)
        else:
            root = os.path.join(args.output_root, f"{now_timestamp()}_single_runs")
            ensure_dir(root)
        out = run_single(root, args.experiment, args.seed)
        print(f"Run completed: {out['run_dir']}")
        return

    if args.cmd == "summarize":
        summarize(args.experiment_root)
        print(f"Summary written to: {args.experiment_root}")
        return


if __name__ == "__main__":
    main()
