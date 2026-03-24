import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from train import get_transforms


import os


def _infer_num_classes(model, fallback=10):
    if hasattr(model, "fc2"):
        return model.fc2.out_features
    if hasattr(model, "linear"):
        return model.linear.out_features
    if hasattr(model, "fc"):
        return model.fc.out_features
    return fallback


def _register_penultimate_hook(model):
    holder = {"feat": None}
    target = None
    if hasattr(model, "fc2"):
        target = model.fc2
    elif hasattr(model, "linear"):
        target = model.linear
    elif hasattr(model, "fc"):
        target = model.fc

    if target is None:
        return None, holder

    def hook(module, inputs, output):
        holder["feat"] = inputs[0].detach()

    handle = target.register_forward_hook(hook)
    return handle, holder


@torch.no_grad()
def evaluate_model(
    model,
    config,
    device,
    checkpoint_path=None,
    collect_predictions=False,
    collect_features=False,
    custom_transform=None,
):
    tf = custom_transform if custom_transform is not None else get_transforms(config, train=False)

    if config.dataset == "mnist":
        test_ds = datasets.MNIST(config.data_dir, train=False, download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=tf)

    loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    num_classes = _infer_num_classes(model, fallback=config.num_classes)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    correct, n = 0, 0

    pred_chunks = {"labels": [], "preds": [], "logits": [], "probs": []} if collect_predictions else None
    hook_handle, feat_holder = _register_penultimate_hook(model) if collect_features else (None, None)
    feat_chunks = {"features": [], "labels": []} if collect_features else None

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)

        correct += preds.eq(labels).sum().item()
        n += imgs.size(0)

        for p, t in zip(preds, labels):
            class_correct[t.item()] += int((p == t).item())
            class_total[t.item()] += 1

        if collect_predictions:
            pred_chunks["labels"].append(labels.detach().cpu().numpy())
            pred_chunks["preds"].append(preds.detach().cpu().numpy())
            pred_chunks["logits"].append(logits.detach().cpu().numpy())
            pred_chunks["probs"].append(probs.detach().cpu().numpy())

        if collect_features and feat_holder and feat_holder["feat"] is not None:
            feat = feat_holder["feat"].detach().cpu().numpy()
            feat_chunks["features"].append(feat)
            feat_chunks["labels"].append(labels.detach().cpu().numpy())

    if hook_handle is not None:
        hook_handle.remove()

    overall = correct / n if n else 0.0
    class_accuracy = []
    for i in range(num_classes):
        denom = class_total[i]
        class_accuracy.append((class_correct[i] / denom) if denom else 0.0)

    result = {
        "overall_accuracy": float(overall),
        "correct": int(correct),
        "total": int(n),
        "class_accuracy": class_accuracy,
        "class_correct": class_correct,
        "class_total": class_total,
    }

    if collect_predictions:
        result["predictions"] = {
            "labels": np.concatenate(pred_chunks["labels"]) if pred_chunks["labels"] else np.array([]),
            "preds": np.concatenate(pred_chunks["preds"]) if pred_chunks["preds"] else np.array([]),
            "logits": np.concatenate(pred_chunks["logits"]) if pred_chunks["logits"] else np.array([]),
            "probs": np.concatenate(pred_chunks["probs"]) if pred_chunks["probs"] else np.array([]),
        }

    if collect_features:
        result["features"] = {
            "features": np.concatenate(feat_chunks["features"]) if feat_chunks["features"] else np.array([]),
            "labels": np.concatenate(feat_chunks["labels"]) if feat_chunks["labels"] else np.array([]),
        }

    return result


@torch.no_grad()
def run_test(model, config, device):
    save_path = os.path.join(config.checkpoint_dir, config.save_path)
    if not os.path.exists(save_path):
        # Fallback to local if not in hidden folder yet
        save_path = config.save_path

    result = evaluate_model(model, config, device, checkpoint_path=save_path, collect_predictions=False, collect_features=False)

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {result['overall_accuracy']:.4f}  ({result['correct']}/{result['total']})\n")
    for i, acc in enumerate(result["class_accuracy"]):
        print(f"  Class {i}: {acc:.4f}  ({result['class_correct'][i]}/{result['class_total'][i]})")
