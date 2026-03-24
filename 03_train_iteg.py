# 03_train_iteg.py  v2.0
"""
Program 3/6: Train ITEG Explainer via Game-Theoretic Optimization
==================================================================
Trains the Explainer network in a two-player zero-sum game against
an Adversary, using Detector A as the fixed black-box model.

Game components (from the paper's payoff function):
  - Faithfulness: MSE between D(X) and D(X*M) softmax probabilities
  - Sparsity: L1 norm of mask M
  - Stability: MSE between M(X) and M(X+delta) under adversarial perturbation

Recovery:
  - Checkpoints every 5 epochs to Drive
  - Resumes from last checkpoint after disconnect

Usage in Colab:
    %run 03_train_iteg.py
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gc

# ============================================================
# CONFIG
# ============================================================
DRIVE_BASE = "/content/drive/MyDrive/ASVspoof_Project"
DATA_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/data")
MODEL_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ITEG hyperparameters
ALPHA = 50.0      # Faithfulness weight (increased from 10 to prevent mask collapse)
BETA = 0.05       # Sparsity weight (reduced from 0.5 — was too aggressive)
GAMMA = 5.0       # Stability weight
EPSILON = 0.05    # Adversary perturbation bound (l2)

# Training config
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 60
CHECKPOINT_EVERY = 5

# Load data config
with open(os.path.join(DATA_DIR, "config.json")) as f:
    data_config = json.load(f)
N_MELS = data_config["n_mels"]
MAX_FRAMES = data_config["max_frames"]


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class DetectorA(nn.Module):
    """Detector A — must match architecture from Program 2."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


class Explainer(nn.Module):
    """
    Explainer network: encoder-decoder that produces a soft mask
    M in [0,1]^{T x F} over the mel-spectrogram.
    Input: (batch, 1, 80, 400)
    Output: (batch, 1, 80, 400) soft mask via sigmoid
    """
    def __init__(self):
        super().__init__()
        # Encoder: downsample to capture broad spectral patterns
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        # Decoder: upsample back to input resolution
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.dec1 = nn.Conv2d(16, 1, 3, padding=1)  # Output: 1 channel mask

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Decoder with skip connections (additive)
        d3 = self.dec3(e3) + e2
        d2 = self.dec2(d3) + e1
        mask = torch.sigmoid(self.dec1(d2))

        return mask


# ============================================================
# DATA LOADING
# ============================================================

def load_chunked_specs(prefix, data_dir):
    """Load and concatenate chunked spectrograms."""
    meta_path = os.path.join(data_dir, f"{prefix}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    chunks = []
    for i in range(meta['n_chunks']):
        chunk = np.load(os.path.join(data_dir, f"{prefix}_specs_chunk{i}.npy"))
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def load_data():
    """Load train spectrograms, normalize, create loader."""
    print("Loading training data...")
    t0 = time.time()

    train_specs = load_chunked_specs("train", DATA_DIR)
    train_labels = np.load(os.path.join(DATA_DIR, "train_labels.npy"))

    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))
    train_specs = (train_specs - stats['mean']) / (stats['std'] + 1e-8)
    train_specs = train_specs[:, np.newaxis, :, :]

    X_train = torch.tensor(train_specs, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    # Use only spoof samples for ITEG training (we want to explain
    # what makes the detector flag something as spoof)
    # Also include some bonafide for balance
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)

    print(f"  Train: {X_train.shape} loaded in {time.time()-t0:.1f}s")
    return train_loader


# ============================================================
# ITEG GAME TRAINING
# ============================================================

def adversarial_perturbation(explainer, X, epsilon):
    """
    Generate adversarial perturbation via single PGD step.
    The adversary maximizes ||M(X) - M(X+delta)|| subject to ||delta||_2 <= epsilon.
    """
    X_adv = X.clone().detach().requires_grad_(True)
    mask = explainer(X_adv)
    # Maximize mask change = minimize negative of mask sum (proxy for gradient direction)
    loss = mask.sum()
    grad = torch.autograd.grad(loss, X_adv, create_graph=False)[0]

    # PGD step: move in gradient direction, then project onto l2 ball
    perturbation = epsilon * grad / (grad.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8)

    return (X + perturbation).detach()


def compute_iteg_loss(detector, explainer, X, epsilon):
    """
    Compute the ITEG composite payoff:
      L = alpha * faithfulness + beta * sparsity + gamma * stability
    """
    # Generate explanation mask
    mask = explainer(X)

    # 1. FAITHFULNESS: D(X) ≈ D(X*M)
    with torch.no_grad():
        orig_logits = detector(X)
        orig_probs = torch.softmax(orig_logits, dim=1)

    masked_input = X * mask
    masked_logits = detector(masked_input)
    masked_probs = torch.softmax(masked_logits, dim=1)

    faithfulness_loss = F.mse_loss(masked_probs, orig_probs)

    # 2. SPARSITY: L1 norm of mask (encourage sparse masks)
    sparsity_loss = torch.mean(torch.abs(mask))

    # 3. STABILITY: ||M(X) - M(X+delta)||_2 under adversarial perturbation
    X_adv = adversarial_perturbation(explainer, X, epsilon)
    mask_adv = explainer(X_adv)
    stability_loss = F.mse_loss(mask, mask_adv)

    # Composite payoff
    total_loss = (ALPHA * faithfulness_loss +
                  BETA * sparsity_loss +
                  GAMMA * stability_loss)

    return total_loss, faithfulness_loss.item(), sparsity_loss.item(), stability_loss.item()


def train_iteg():
    """Main ITEG training loop with checkpointing."""
    final_path = os.path.join(MODEL_DIR, "explainer_final.pth")

    # Skip if already done
    if os.path.exists(final_path):
        print("\n  [SKIP] Explainer already trained. Loading...")
        explainer = Explainer().to(DEVICE)
        explainer.load_state_dict(torch.load(final_path, map_location=DEVICE,
                                              weights_only=True))
        return explainer

    # Load detector A (frozen)
    print("Loading Detector A (frozen)...")
    detector = DetectorA().to(DEVICE)
    detector.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "detector_A_final.pth"),
        map_location=DEVICE, weights_only=True))
    detector.eval()
    for p in detector.parameters():
        p.requires_grad = False
    print("  Detector A loaded and frozen.")

    # Load data
    train_loader = load_data()

    # Initialize explainer
    explainer = Explainer().to(DEVICE)
    optimizer = optim.Adam(explainer.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    start_epoch = 0
    loss_history = []

    # Resume from checkpoint
    ckpt_path = os.path.join(MODEL_DIR, "explainer_checkpoint.pth")
    if os.path.exists(ckpt_path):
        print("\n  [RESUME] Loading explainer checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        explainer.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt.get('loss_history', [])
        print(f"    Resuming from epoch {start_epoch}")
    else:
        print("\n  Training ITEG Explainer from scratch...")

    param_count = sum(p.numel() for p in explainer.parameters())
    print(f"  Explainer parameters: {param_count:,}")
    print(f"  Config: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, "
          f"epsilon={EPSILON}")
    print(f"  Epochs: {start_epoch} -> {NUM_EPOCHS}, "
          f"batch={BATCH_SIZE}, lr={LEARNING_RATE}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        explainer.train()
        epoch_loss = 0.0
        epoch_faith = 0.0
        epoch_sparse = 0.0
        epoch_stab = 0.0
        n_batches = 0
        t0 = time.time()

        for X, y in train_loader:
            X = X.to(DEVICE)

            optimizer.zero_grad()
            loss, faith, sparse, stab = compute_iteg_loss(
                detector, explainer, X, EPSILON)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_faith += faith
            epoch_sparse += sparse
            epoch_stab += stab
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        avg_faith = epoch_faith / n_batches
        avg_sparse = epoch_sparse / n_batches
        avg_stab = epoch_stab / n_batches
        epoch_time = time.time() - t0

        loss_history.append({
            'epoch': epoch + 1,
            'total': avg_loss,
            'faithfulness': avg_faith,
            'sparsity': avg_sparse,
            'stability': avg_stab
        })

        # Print every epoch (compact)
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"L={avg_loss:.4f} "
              f"(F={avg_faith:.4f} S={avg_sparse:.3f} St={avg_stab:.5f}) | "
              f"{epoch_time:.1f}s")

        # Checkpoint every 5 epochs
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            # Mask health check
            explainer.eval()
            with torch.no_grad():
                for X_check, _ in train_loader:
                    X_check = X_check.to(DEVICE)
                    mask_check = explainer(X_check)
                    m_mean = mask_check.mean().item()
                    m_max = mask_check.max().item()
                    active = (mask_check > 0.5).float().mean().item()
                    print(f"    Mask: mean={m_mean:.4f}, max={m_max:.4f}, "
                          f"active(>0.5)={active:.3f}")
                    if m_max < 0.05:
                        print(f"    [WARN] Mask may be collapsing! "
                              f"max={m_max:.4f}")
                    break
            explainer.train()

            try:
                torch.save({
                    'epoch': epoch,
                    'model_state': explainer.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'loss_history': loss_history,
                }, ckpt_path)
                print(f"    [CHECKPOINT] Saved at epoch {epoch+1}")
            except Exception as e:
                print(f"    [WARN] Checkpoint failed: {e}")

    # Save final model
    torch.save(explainer.state_dict(), final_path)
    print(f"\n  [OK] Explainer saved to {final_path}")

    # Save loss history
    history_path = os.path.join(MODEL_DIR, "iteg_loss_history.json")
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"  Loss history saved to {history_path}")

    # Clean up checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Quick summary of mask statistics on a batch
    explainer.eval()
    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(DEVICE)
            mask = explainer(X)
            print(f"\n  Mask statistics (sample batch):")
            print(f"    Mean: {mask.mean().item():.4f}")
            print(f"    Std:  {mask.std().item():.4f}")
            print(f"    Min:  {mask.min().item():.4f}")
            print(f"    Max:  {mask.max().item():.4f}")
            print(f"    Sparsity (fraction < 0.1): "
                  f"{(mask < 0.1).float().mean().item():.3f}")
            print(f"    Active (fraction > 0.5): "
                  f"{(mask > 0.5).float().mean().item():.3f}")
            break

    return explainer


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 3/6: ITEG TRAINING  v2.0")
    print("=" * 60)

    # Clear GPU memory from any previous runs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    explainer = train_iteg()

    print("\n" + "=" * 60)
    print("ITEG TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModels in: {MODEL_DIR}")
    for f in sorted(os.listdir(MODEL_DIR)):
        fpath = os.path.join(MODEL_DIR, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f:45s} {size_mb:8.2f} MB")
    print(f"\n>>> Ready for Program 4 (Baselines & Evaluation). <<<")


if __name__ == "__main__":
    main()
