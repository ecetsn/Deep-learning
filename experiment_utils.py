import json
import os
import tempfile
import time
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from ptflops import get_model_complexity_info
from torchvision import transforms


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def canonical_run_id(experiment_name: str, seed: int) -> str:
    return f"{experiment_name}__seed{seed}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def save_numpy(path: str, array: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, array)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez(path, **arrays)


def model_complexity(model: torch.nn.Module, input_res=(3, 32, 32)) -> Dict[str, float]:
    macs, params = get_model_complexity_info(
        model,
        input_res,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return {
        "macs": float(macs),
        "mmacs": float(macs / 1e6),
        "params": float(params),
        "mparams": float(params / 1e6),
    }


def best_epoch_and_acc(history: Dict[str, List[float]]) -> Dict[str, Any]:
    val_acc = history.get("val_acc", [])
    if not val_acc:
        return {"best_epoch": None, "best_val_acc": None, "final_val_acc": None}
    best_val_acc = float(max(val_acc))
    best_epoch = int(val_acc.index(max(val_acc)) + 1)
    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": float(val_acc[-1]),
    }


def rows_to_csv(path: str, rows: Iterable[Dict[str, Any]], columns: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            vals = []
            for c in columns:
                v = row.get(c, "")
                if isinstance(v, str):
                    text = v.replace('"', '""')
                    if "," in text or '"' in text:
                        text = f'"{text}"'
                    vals.append(text)
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def build_adam_step_scheduler(params, config):
    optimizer = torch.optim.Adam(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )
    return optimizer, scheduler


def get_transfer_transforms() -> Dict[str, Dict[str, transforms.Compose]]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    option1 = {
        "train": transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]),
    }
    option2 = {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    }
    return {"option1": option1, "option2": option2}
