"""main.py — CS515 HW3 command-line entry point.

Usage examples::

    python main.py train-clean --epochs 30 --lr 0.01
    python main.py train-augmix --epochs 30 --lr 0.01 --jsd
    python main.py eval-clean --ckpt checkpoints/teacher_clean.pth
    python main.py eval-cifar10c --ckpt checkpoints/teacher_clean.pth
    python main.py eval-pgd --ckpt checkpoints/teacher_augmix.pth --norm linf --eps 0.0157
    python main.py distill --teacher checkpoints/teacher_augmix.pth --alpha 0.9 --temperature 4
    python main.py transfer-attack \\
        --teacher checkpoints/teacher_augmix.pth \\
        --student checkpoints/student_augmix.pth

Each subcommand constructs the appropriate ``@dataclass`` config from
``parameters.py`` and delegates to functions in ``train.py`` / ``test.py``.
"""

import argparse
import os
import sys
from typing import Optional

import torch

from parameters import (
    AugMixConfig,
    DistillConfig,
    EvalConfig,
    PGDConfig,
    TrainConfig,
    set_seed,
)


# Model builders

def _build_teacher(num_classes: int = 10) -> torch.nn.Module:
    """Instantiate the teacher: ResNet-18 equivalent for CIFAR-10.

    Args:
        num_classes: Number of output classes.  Default: ``10``.

    Returns:
        An initialised :class:`models.ResNet.ResNet` instance with
        ``BasicBlock`` and ``[2, 2, 2, 2]`` layers.
    """
    from models.ResNet import BasicBlock, ResNet
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def _build_student(num_classes: int = 10) -> torch.nn.Module:
    """Instantiate the compact student CNN.

    Args:
        num_classes: Number of output classes.  Default: ``10``.

    Returns:
        An initialised :class:`models.student.StudentCNN` instance.
    """
    from models.student import StudentCNN
    return StudentCNN(num_classes=num_classes)


def _load_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load model weights from a checkpoint file.

    Args:
        model: An uninitialised model to load weights into.
        ckpt_path: Path to the ``.pth`` checkpoint.
        device: Target compute device.

    Returns:
        The model with loaded weights, moved to ``device``.

    Raises:
        FileNotFoundError: If ``ckpt_path`` does not exist.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)


def _print_complexity(model: torch.nn.Module, label: str) -> None:
    """Print FLOPs and parameter count using ptflops.

    Args:
        model: Neural network to profile.
        label: Human-readable label for the printout.
    """
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, (3, 32, 32),
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
        print(f"[{label}] MACs: {macs}  |  Params: {params}")
    except Exception as exc:
        print(f"[{label}] ptflops failed: {exc}")


# Subcommand handlers

def cmd_train_clean(args: argparse.Namespace) -> None:
    """Handle the ``train-clean`` subcommand."""
    from train import run_training, log_section

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_path=args.save_path or "teacher_clean.pth",
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    if args.device:
        config.device = args.device

    set_seed(config.seed)
    device = torch.device(config.device)
    model  = _build_teacher().to(device)

    print(f"\n{'='*60}")
    print("  train-clean: Clean ResNet-18 teacher on CIFAR-10")
    print(f"{'='*60}")
    _print_complexity(model, "Teacher")

    run_training(model, config, device)


def cmd_train_augmix(args: argparse.Namespace) -> None:
    """Handle the ``train-augmix`` subcommand."""
    from train import run_augmix_training

    config = AugMixConfig(
        jsd=args.jsd,
        mixture_width=args.mixture_width,
        mixture_depth=args.mixture_depth,
        aug_severity=args.aug_severity,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_path=args.save_path or "teacher_augmix.pth",
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    if args.device:
        config.device = args.device

    set_seed(config.seed)
    device = torch.device(config.device)
    model  = _build_teacher().to(device)

    print(f"\n{'='*60}")
    print("  train-augmix: AugMix ResNet-18 teacher on CIFAR-10")
    print(f"  JSD loss: {config.jsd}")
    print(f"{'='*60}")
    _print_complexity(model, "Teacher (AugMix)")

    run_augmix_training(model, config, device)


def cmd_eval_clean(args: argparse.Namespace) -> None:
    """Handle the ``eval-clean`` subcommand."""
    from data_utils import get_cifar10_test_loader
    from test import eval_clean
    from train import log_section

    config = EvalConfig(
        ckpt=args.ckpt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    if args.device:
        config.device = args.device

    device = torch.device(config.device)
    model_fn = _build_student if args.model == "student" else _build_teacher
    model  = _load_model(model_fn(), config.ckpt, device)
    _print_complexity(model, f"Evaluated {args.model}")

    loader = get_cifar10_test_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers
    )
    tag = os.path.splitext(os.path.basename(config.ckpt))[0]
    acc = eval_clean(model, loader, device)
    log_section(
        f"eval-clean — {tag}",
        f"- Checkpoint: `{config.ckpt}`\n"
        f"- **Clean accuracy: {acc*100:.2f}%**",
        results_dir=config.results_dir,
    )


def cmd_eval_cifar10c(args: argparse.Namespace) -> None:
    """Handle the ``eval-cifar10c`` subcommand."""
    from test import eval_cifar10c

    config = EvalConfig(
        ckpt=args.ckpt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    if args.device:
        config.device = args.device

    device = torch.device(config.device)
    model_fn = _build_student if args.model == "student" else _build_teacher
    model  = _load_model(model_fn(), config.ckpt, device)
    tag    = os.path.splitext(os.path.basename(config.ckpt))[0]

    cifar10c_dir = os.path.join(config.data_dir, "CIFAR-10-C")
    eval_cifar10c(
        model, cifar10c_dir, config.mean, config.std, device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        results_dir=config.results_dir,
        ckpt_tag=f"{args.model}_{tag}",
    )


def cmd_eval_pgd(args: argparse.Namespace) -> None:
    """Handle the ``eval-pgd`` subcommand."""
    from data_utils import get_cifar10_test_loader
    from test import eval_pgd, run_gradcam, run_tsne

    config = PGDConfig(
        norm=args.norm,
        eps=args.eps,
        alpha=args.alpha if args.alpha is not None else args.eps / 4.0,
        n_iter=args.n_iter,
        random_start=not args.no_random_start,
        ckpt=args.ckpt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    if args.device:
        config.device = args.device

    device = torch.device(config.device)
    model_fn = _build_student if args.model == "student" else _build_teacher
    model  = _load_model(model_fn(), config.ckpt, device)
    tag    = os.path.splitext(os.path.basename(config.ckpt))[0]
    tag    = f"{args.model}_{tag}"

    loader = get_cifar10_test_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers
    )

    # PGD evaluation
    eval_pgd(
        model, loader, device,
        norm=config.norm, eps=config.eps, alpha=config.alpha,
        n_iter=config.n_iter, random_start=config.random_start,
        results_dir=config.results_dir, ckpt_tag=tag,
    )

    # Grad-CAM (always run for the L∞ variant)
    if args.gradcam and config.norm == "linf":
        # target_layer selection
        if args.model == "teacher":
            target_layer = model.layer4
        else:
            # For StudentCNN, use the last conv block in features
            target_layer = model.features[-2] # Before ReLU of last conv
        run_gradcam(
            model, loader, device, target_layer,
            mean=config.mean, std=config.std,
            norm="linf", eps=config.eps, alpha=config.alpha, n_iter=config.n_iter,
            n_samples=2, results_dir=config.results_dir, ckpt_tag=tag,
        )

    # t-SNE (always run for the L∞ variant)
    if args.tsne and config.norm == "linf":
        hook_layer = model.avgpool
        run_tsne(
            model, loader, device, hook_layer,
            norm="linf", eps=config.eps, alpha=config.alpha, n_iter=config.n_iter,
            n_samples=1000, results_dir=config.results_dir, ckpt_tag=tag,
        )


def cmd_distill(args: argparse.Namespace) -> None:
    """Handle the ``distill`` subcommand."""
    from train import run_distillation

    config = DistillConfig(
        teacher_ckpt=args.teacher,
        alpha=args.alpha,
        temperature=args.temperature,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        save_path=args.save_path or "student_augmix.pth",
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    if args.device:
        config.device = args.device

    set_seed(config.seed)
    device  = torch.device(config.device)
    teacher = _load_model(_build_teacher(), config.teacher_ckpt, device)
    student = _build_student().to(device)

    print(f"\n{'='*60}")
    print(f"  distill: α={config.alpha}  T={config.temperature}")
    print(f"{'='*60}")
    _print_complexity(teacher, "Teacher")
    _print_complexity(student, "Student")

    run_distillation(student, teacher, config, device)


def cmd_transfer_attack(args: argparse.Namespace) -> None:
    """Handle the ``transfer-attack`` subcommand."""
    from data_utils import get_cifar10_test_loader
    from test import eval_transfer_attack

    config = EvalConfig(
        teacher_ckpt=args.teacher,
        student_ckpt=args.student,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        num_workers=args.num_workers,
    )
    if args.device:
        config.device = args.device

    device  = torch.device(config.device)
    teacher = _load_model(_build_teacher(), config.teacher_ckpt, device)
    student = _load_model(_build_student(), config.student_ckpt, device)

    loader = get_cifar10_test_loader(
        config.data_dir, config.batch_size, config.mean, config.std,
        config.num_workers
    )

    eps   = args.eps
    alpha = args.alpha if args.alpha is not None else eps / 4.0
    eval_transfer_attack(
        teacher, student, loader, device,
        eps=eps, alpha=alpha, n_iter=args.n_iter,
        results_dir=config.results_dir,
    )


# Argument parser

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add hyperparameters shared by multiple subcommands.

    Args:
        parser: The subparser to extend.
    """
    parser.add_argument("--data_dir",      type=str,   default="./data")
    parser.add_argument("--batch_size",    type=int,   default=128)
    parser.add_argument("--device",        type=str,   default=None,
                        help="Override device (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument("--results_dir",   type=str,   default="results")
    parser.add_argument("--num_workers",   type=int,   default=2)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific hyperparameters.

    Args:
        parser: The subparser to extend.
    """
    parser.add_argument("--epochs",               type=int,   default=30)
    parser.add_argument("--lr",                   type=float, default=0.01)
    parser.add_argument("--weight_decay",         type=float, default=5e-4)
    parser.add_argument("--label_smoothing",      type=float, default=0.0)
    parser.add_argument("--scheduler_step_size",  type=int,   default=10)
    parser.add_argument("--scheduler_gamma",      type=float, default=0.1)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--checkpoint_dir",       type=str,   default="checkpoints")
    parser.add_argument("--save_path",            type=str,   default=None)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="CS515 HW3 — Robustness, AugMix, PGD, Distillation",
    )
    subs = parser.add_subparsers(dest="subcommand", required=True)

    # ---- train-clean ----
    p_tc = subs.add_parser("train-clean", help="Clean training of teacher ResNet")
    _add_common_args(p_tc)
    _add_train_args(p_tc)
    p_tc.set_defaults(func=cmd_train_clean)

    # ---- train-augmix ----
    p_ta = subs.add_parser("train-augmix", help="AugMix training of teacher ResNet")
    _add_common_args(p_ta)
    _add_train_args(p_ta)
    p_ta.add_argument("--jsd",            action="store_true",
                      help="Use JSD consistency loss (recommended)")
    p_ta.add_argument("--mixture_width",  type=int,   default=3)
    p_ta.add_argument("--mixture_depth",  type=int,   default=-1,
                      help="Augmentation chain depth; -1 = random in {1,2,3}")
    p_ta.add_argument("--aug_severity",   type=int,   default=3)
    p_ta.set_defaults(func=cmd_train_augmix)

    # ---- eval-clean ----
    p_ec = subs.add_parser("eval-clean", help="Evaluate model on clean CIFAR-10 test set")
    _add_common_args(p_ec)
    p_ec.add_argument("--ckpt", type=str, required=True)
    p_ec.add_argument("--model", type=str, default="teacher", choices=["teacher", "student"])
    p_ec.set_defaults(func=cmd_eval_clean)

    # ---- eval-cifar10c ----
    p_c10c = subs.add_parser("eval-cifar10c",
                              help="Evaluate on CIFAR-10-C (15 corruptions × 5 severities)")
    _add_common_args(p_c10c)
    p_c10c.add_argument("--ckpt", type=str, required=True)
    p_c10c.add_argument("--model", type=str, default="teacher", choices=["teacher", "student"])
    p_c10c.set_defaults(func=cmd_eval_cifar10c)

    # ---- eval-pgd ----
    p_pgd = subs.add_parser("eval-pgd",
                             help="PGD-20 adversarial evaluation + Grad-CAM + t-SNE")
    _add_common_args(p_pgd)
    p_pgd.add_argument("--ckpt",            type=str,   required=True)
    p_pgd.add_argument("--model",           type=str,   default="teacher", choices=["teacher", "student"])
    p_pgd.add_argument("--norm",            type=str,   default="linf",
                       choices=["linf", "l2"])
    p_pgd.add_argument("--eps",             type=float, default=4.0/255.0)
    p_pgd.add_argument("--alpha",           type=float, default=None,
                       help="Per-step size; defaults to eps/4")
    p_pgd.add_argument("--n_iter",          type=int,   default=20)
    p_pgd.add_argument("--no_random_start", action="store_true")
    p_pgd.add_argument("--gradcam",         action="store_true",
                       help="Also generate Grad-CAM heatmaps (linf only)")
    p_pgd.add_argument("--tsne",            action="store_true",
                       help="Also generate t-SNE embedding (linf only)")
    p_pgd.set_defaults(func=cmd_eval_pgd)

    # ---- distill ----
    p_dist = subs.add_parser("distill", help="Hinton KD: teacher → student")
    _add_common_args(p_dist)
    p_dist.add_argument("--teacher",             type=str,   required=True)
    p_dist.add_argument("--alpha",               type=float, default=0.9)
    p_dist.add_argument("--temperature",         type=float, default=4.0)
    p_dist.add_argument("--epochs",              type=int,   default=30)
    p_dist.add_argument("--lr",                  type=float, default=0.01)
    p_dist.add_argument("--weight_decay",        type=float, default=5e-4)
    p_dist.add_argument("--seed",                type=int,   default=42)
    p_dist.add_argument("--checkpoint_dir",      type=str,   default="checkpoints")
    p_dist.add_argument("--save_path",           type=str,   default=None)
    p_dist.set_defaults(func=cmd_distill)

    # ---- transfer-attack ----
    p_tr = subs.add_parser("transfer-attack",
                            help="Evaluate adversarial transferability (teacher → student)")
    _add_common_args(p_tr)
    p_tr.add_argument("--teacher",  type=str,   required=True)
    p_tr.add_argument("--student",  type=str,   required=True)
    p_tr.add_argument("--eps",      type=float, default=4.0/255.0)
    p_tr.add_argument("--alpha",    type=float, default=None)
    p_tr.add_argument("--n_iter",   type=int,   default=20)
    p_tr.set_defaults(func=cmd_transfer_attack)

    return parser


# Entry point

def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand handler."""
    parser = build_parser()
    args   = parser.parse_args()

    # Ensure results/ exists for all subcommands
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch: {torch.__version__}")
    args.func(args)


if __name__ == "__main__":
    main()