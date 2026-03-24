import torch
import torch.nn as nn
import torch.nn.functional as F
from train import validate, get_loaders, save_history
from ptflops import get_model_complexity_info
from experiment_utils import build_adam_step_scheduler
import os

class KDLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(KDLoss, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Standard KD Loss
        soft_targets = F.log_softmax(student_logits / self.temp, dim=1)
        soft_probs = F.softmax(teacher_logits / self.temp, dim=1)
        
        distillation_loss = self.kl_div(soft_targets, soft_probs) * (self.temp ** 2)
        student_loss = self.ce(student_logits, labels)
        
        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss

class CustomKDLoss(nn.Module):
    """
    As specified in the assignment:
    "use the outputs of the teacher model to assign a probability to true class only 
    (other classes will be assigned equal probability), this way we assign a certain 
    difficulty to the examples"
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        super(CustomKDLoss, self).__init__()
        self.temp = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Teacher's probability for true class
        teacher_probs = F.softmax(teacher_logits, dim=1)
        batch_size, num_classes = teacher_probs.shape
        
        # Extract true class probability
        true_class_probs = teacher_probs[torch.arange(batch_size), labels] # [B]
        
        # Create new target distribution: 
        # P(true) = teacher's true class prob
        # P(others) = (1 - P(true)) / (num_classes - 1)
        custom_targets = torch.full_like(teacher_probs, 0.0)
        remaining_prob = (1.0 - true_class_probs) / (num_classes - 1)
        
        for i in range(num_classes):
            custom_targets[:, i] = remaining_prob
        
        custom_targets[torch.arange(batch_size), labels] = true_class_probs
        
        # Loss
        soft_targets = F.log_softmax(student_logits / self.temp, dim=1)
        # Note: custom_targets are already probabilities, we treat them as fixed "soft targets"
        distillation_loss = self.kl_div(soft_targets, custom_targets) * (self.temp ** 2)
        student_loss = self.ce(student_logits, labels)
        
        return self.alpha * student_loss + (1 - self.alpha) * distillation_loss

def track_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    return entropy.mean().item()

def train_kd(student, teacher, loader, optimizer, criterion, device, log_interval):
    student.train()
    teacher.eval() # Teacher is always in eval mode
    total_loss, correct, n = 0.0, 0, 0
    total_entropy = 0.0
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        with torch.no_grad():
            teacher_logits = teacher(imgs)
            total_entropy += track_entropy(teacher_logits)
            
        optimizer.zero_grad()
        student_logits = student(imgs)
        loss = criterion(student_logits, teacher_logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach().item() * imgs.size(0)
        correct += student_logits.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)
        
        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")
            
    return total_loss / n, correct / n, total_entropy / len(loader)

def run_kd_experiment(name, student, teacher, config, custom=False, out_dir=None):
    if config.epochs < 1:
        raise ValueError("config.epochs must be >= 1")

    print(f"\n--- Running KD Experiment: {name} ---")
    device = torch.device(config.device)
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    
    # Check complexities
    s_macs, s_params = get_model_complexity_info(student, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    t_macs, t_params = get_model_complexity_info(teacher, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    print(f"Teacher complexity: {t_macs} MACs, {t_params} params")
    print(f"Student complexity: {s_macs} MACs, {s_params} params")
    
    train_loader, val_loader = get_loaders(config)
    optimizer, scheduler = build_adam_step_scheduler(student.parameters(), config)
    
    if custom:
        criterion = CustomKDLoss(temperature=config.temperature, alpha=config.alpha)
    else:
        criterion = KDLoss(temperature=config.temperature, alpha=config.alpha)
        
    best_acc = float("-inf")
    history = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": [], 
        "lr": [],
        "teacher_entropy": []
    }
    target_dir = out_dir or config.checkpoint_dir
    os.makedirs(target_dir, exist_ok=True)
    history_name = f"student_{name}_history.json"
    save_path = os.path.join(target_dir, f"student_{name}_best.pth")

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc, avg_ent = train_kd(student, teacher, train_loader, optimizer, criterion, device, config.log_interval)
        val_loss, val_acc = validate(student, val_loader, nn.CrossEntropyLoss(), device)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Val Acc {val_acc:.4f}, Teacher Avg Entropy: {avg_ent:.4f}, LR {current_lr:.6f}")
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["teacher_entropy"].append(avg_ent)
        scheduler.step()
        
        # Incremental save
        save_history(history, config, name=history_name, out_dir=target_dir)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), save_path)
            print(f"  Saved best student model (val_acc={best_acc:.4f}) to {save_path}")
            
    return {
        "best_acc": float(best_acc),
        "student_macs_str": s_macs,
        "history": history,
        "save_path": save_path,
        "history_path": os.path.join(target_dir, history_name),
    }

if __name__ == "__main__":
    raise RuntimeError(
        "Direct execution is disabled. Use `python experiment.py run-suite --suite hw1b` "
        "or `python experiment.py run-single ...`."
    )
