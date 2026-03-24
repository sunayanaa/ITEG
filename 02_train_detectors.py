# 02_train_detectors.py  v2.0
"""
Program 2/6: Train Two CNN Detectors for ITEG Experiments
==========================================================
v2.0 Changes:
  - Simpler architectures (less overfitting on 10K samples)
  - Class-weighted CrossEntropy (handles 75/25 spoof/bonafide imbalance)
  - No dropout in conv layers
  - Cosine annealing LR schedule
  - Early stopping with patience
  - Detector A: 3-conv + GAP (target)
  - Detector B: 2-conv + GAP with different kernel sizes (transfer)

Recovery:
  - Checkpoints every 5 epochs to Drive
  - Resumes from last checkpoint after disconnect
  - Skips if final model exists

Usage in Colab:
    %run 02_train_detectors.py
"""

import os
import sys
import json
import time
import numpy as np

# ============================================================
# MOUNT DRIVE
# ============================================================
try:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted.")
    else:
        print("Google Drive already mounted.")
except ImportError:
    print("Not running in Colab — skipping drive mount.")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# CONFIG
# ============================================================
DRIVE_BASE = "/content/drive/MyDrive/ASVspoof_Project"
DATA_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/data")
MODEL_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50
CHECKPOINT_EVERY = 5
PATIENCE = 10  # Early stopping patience

# Load data config
with open(os.path.join(DATA_DIR, "config.json")) as f:
    data_config = json.load(f)
N_MELS = data_config["n_mels"]
MAX_FRAMES = data_config["max_frames"]


# ============================================================
# DETECTOR ARCHITECTURES
# ============================================================

class DetectorA(nn.Module):
    """
    Detector A (Target): 3-conv layers, moderate capacity.
    Input: (batch, 1, 80, 400)
    Output: (batch, 2) logits
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),              # -> (16, 40, 200)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),              # -> (32, 20, 100)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),      # -> (64, 1, 1)
            nn.Flatten()                  # -> (64,)
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


class DetectorB(nn.Module):
    """
    Detector B (Transfer): 2-conv layers, different kernel sizes.
    Deliberately different architecture for transferability test.
    Input: (batch, 1, 80, 400)
    Output: (batch, 2) logits
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),              # -> (16, 40, 200)

            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),      # -> (32, 1, 1)
            nn.Flatten()                  # -> (32,)
        )
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# DATA LOADING
# ============================================================

def load_chunked_specs(prefix, data_dir):
    """Load and concatenate chunked spectrograms from Drive."""
    meta_path = os.path.join(data_dir, f"{prefix}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    chunks = []
    for i in range(meta['n_chunks']):
        chunk = np.load(os.path.join(data_dir, f"{prefix}_specs_chunk{i}.npy"))
        chunks.append(chunk)
        print(f"    Loaded {prefix} chunk {i}: {chunk.shape}")
    return np.concatenate(chunks, axis=0)


def load_data():
    """Load spectrograms, normalize, compute class weights, create loaders."""
    print("Loading data from Drive...")
    t0 = time.time()

    train_specs = load_chunked_specs("train", DATA_DIR)
    train_labels = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
    eval_specs = load_chunked_specs("eval", DATA_DIR)
    eval_labels = np.load(os.path.join(DATA_DIR, "eval_labels.npy"))

    # Normalization
    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))
    mean_val, std_val = stats['mean'], stats['std']
    train_specs = (train_specs - mean_val) / (std_val + 1e-8)
    eval_specs = (eval_specs - mean_val) / (std_val + 1e-8)

    # Add channel dim: (N, 80, 400) -> (N, 1, 80, 400)
    train_specs = train_specs[:, np.newaxis, :, :]
    eval_specs = eval_specs[:, np.newaxis, :, :]

    # Compute class weights for imbalanced data
    n_bonafide = np.sum(train_labels == 0)
    n_spoof = np.sum(train_labels == 1)
    total = n_bonafide + n_spoof
    weight_bonafide = total / (2.0 * n_bonafide)
    weight_spoof = total / (2.0 * n_spoof)
    class_weights = torch.tensor([weight_bonafide, weight_spoof],
                                  dtype=torch.float32).to(DEVICE)
    print(f"  Class weights: bonafide={weight_bonafide:.2f}, "
          f"spoof={weight_spoof:.2f}")

    # Convert to tensors
    X_train = torch.tensor(train_specs, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_eval = torch.tensor(eval_specs, dtype=torch.float32)
    y_eval = torch.tensor(eval_labels, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    eval_loader = DataLoader(TensorDataset(X_eval, y_eval),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    elapsed = time.time() - t0
    print(f"  Train: {X_train.shape}, Eval: {X_eval.shape}")
    print(f"  Loaded in {elapsed:.1f}s")

    return train_loader, eval_loader, class_weights


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion):
    """Compute accuracy and loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            n_batches += 1
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total, total_loss / n_batches


def compute_eer(model, loader):
    """Compute Equal Error Rate."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            out = model(X)
            probs = torch.softmax(out, dim=1)[:, 1]  # P(spoof)
            all_scores.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # Sweep thresholds
    thresholds = np.linspace(0, 1, 2000)
    min_diff = float('inf')
    eer = 0.5

    for t in thresholds:
        preds = (scores >= t).astype(int)
        bonafide_mask = labels == 0
        spoof_mask = labels == 1

        far = np.sum(preds[bonafide_mask] == 1) / max(np.sum(bonafide_mask), 1)
        frr = np.sum(preds[spoof_mask] == 0) / max(np.sum(spoof_mask), 1)

        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0

    return eer


# ============================================================
# TRAINING
# ============================================================

def get_checkpoint_path(name):
    return os.path.join(MODEL_DIR, f"{name}_checkpoint.pth")

def get_final_path(name):
    return os.path.join(MODEL_DIR, f"{name}_final.pth")


def train_detector(model, name, train_loader, eval_loader, class_weights):
    """Train with checkpointing, early stopping, and cosine LR."""
    final_path = get_final_path(name)

    # Skip if already done
    if os.path.exists(final_path):
        print(f"\n  [SKIP] {name}: loading final model...")
        model.load_state_dict(torch.load(final_path, map_location=DEVICE,
                                          weights_only=True))
        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        acc, loss = evaluate(model, eval_loader, criterion)
        eer = compute_eer(model, eval_loader)
        print(f"    Eval — Acc: {acc*100:.2f}%, Loss: {loss:.4f}, EER: {eer*100:.2f}%")
        return model

    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    start_epoch = 0
    best_eval_acc = 0.0
    patience_counter = 0

    # Resume from checkpoint
    ckpt_path = get_checkpoint_path(name)
    if os.path.exists(ckpt_path):
        print(f"\n  [RESUME] {name}: loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_eval_acc = ckpt.get('best_eval_acc', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        print(f"    Resuming from epoch {start_epoch}, "
              f"best_acc={best_eval_acc*100:.1f}%")
    else:
        print(f"\n  Training {name} from scratch...")

    print(f"  Config: epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, "
          f"lr={LEARNING_RATE}, patience={PATIENCE}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        # Evaluate every 5 epochs or last epoch
        if (epoch + 1) % CHECKPOINT_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            eval_acc, eval_loss = evaluate(model, eval_loader, criterion)
            print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Train: {train_acc*100:.1f}% ({train_loss:.4f}) | "
                  f"Eval: {eval_acc*100:.1f}% ({eval_loss:.4f}) | "
                  f"lr={current_lr:.6f} | {epoch_time:.1f}s")

            # Track best model
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                patience_counter = 0
                # Save best model state
                torch.save(model.state_dict(),
                           os.path.join(MODEL_DIR, f"{name}_best.pth"))
            else:
                patience_counter += CHECKPOINT_EVERY

            # Save checkpoint
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_eval_acc': best_eval_acc,
                    'patience_counter': patience_counter,
                }, ckpt_path)
            except Exception as e:
                print(f"    [WARN] Checkpoint save failed: {e}")

            # Early stopping
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break
        else:
            print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                  f"Train: {train_acc*100:.1f}% ({train_loss:.4f}) | "
                  f"lr={current_lr:.6f} | {epoch_time:.1f}s")

    # Load best model for final save
    best_path = os.path.join(MODEL_DIR, f"{name}_best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                          weights_only=True))
        print(f"  Loaded best model (acc={best_eval_acc*100:.1f}%)")

    # Save final
    torch.save(model.state_dict(), final_path)
    print(f"  [OK] {name} saved to {final_path}")

    # Clean up checkpoint and best
    for p in [ckpt_path, best_path]:
        if os.path.exists(p):
            os.remove(p)

    # Final eval with EER
    eval_acc, eval_loss = evaluate(model, eval_loader, criterion)
    eer = compute_eer(model, eval_loader)
    print(f"  Final — Acc: {eval_acc*100:.2f}%, "
          f"Loss: {eval_loss:.4f}, EER: {eer*100:.2f}%")

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 2/6: DETECTOR TRAINING  v2.0")
    print("=" * 60)

    # Delete old v1.0 models if they exist
    for old_file in ["detector_A_final.pth", "detector_B_final.pth"]:
        old_path = os.path.join(MODEL_DIR, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"  Removed old model: {old_file}")

    train_loader, eval_loader, class_weights = load_data()

    # Train Detector A
    print("\n" + "-" * 40)
    print("DETECTOR A (Target, 3-conv CNN)")
    print("-" * 40)
    detector_A = DetectorA()
    param_count_A = sum(p.numel() for p in detector_A.parameters())
    print(f"  Parameters: {param_count_A:,}")
    detector_A = train_detector(detector_A, "detector_A",
                                train_loader, eval_loader, class_weights)

    # Train Detector B
    print("\n" + "-" * 40)
    print("DETECTOR B (Transfer, 2-conv CNN)")
    print("-" * 40)
    detector_B = DetectorB()
    param_count_B = sum(p.numel() for p in detector_B.parameters())
    print(f"  Parameters: {param_count_B:,}")
    detector_B = train_detector(detector_B, "detector_B",
                                train_loader, eval_loader, class_weights)

    # Summary
    print("\n" + "=" * 60)
    print("DETECTOR TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModels in: {MODEL_DIR}")
    for f in sorted(os.listdir(MODEL_DIR)):
        fpath = os.path.join(MODEL_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f:40s} {size_mb:8.2f} MB")
    print(f"\n>>> Ready for Program 3 (ITEG Training). <<<")


if __name__ == "__main__":
    main()
