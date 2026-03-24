# 03b_ablation.py  v1.0
"""
Program 3b: ITEG Ablation Study
=================================
Trains two ITEG variants to isolate the contribution of each
payoff component:
  1. No Stability (gamma=0): Explainer without adversary
  2. No Sparsity (beta=0): Explainer without entropy penalty

Then evaluates all three (+ full ITEG) on faithfulness, sparsity,
stability. Results saved as JSON for the paper's ablation table.

Recovery: Each variant checkpoints every 5 epochs. Skips if final
model exists. Evaluation skips if results file exists.

Usage in Colab:
    %run 03b_ablation.py
"""

import os
import sys
import json
import time
import numpy as np
import gc

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

# ============================================================
# CONFIG
# ============================================================
DRIVE_BASE = "/content/drive/MyDrive/ASVspoof_Project"
DATA_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/data")
MODEL_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/models")
RESULTS_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Shared training config
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 60
CHECKPOINT_EVERY = 5
EPSILON = 0.05

# Ablation variants
VARIANTS = {
    "no_stability": {"alpha": 50.0, "beta": 0.05, "gamma": 0.0,
                     "desc": "No Stability (gamma=0)"},
    "no_sparsity":  {"alpha": 50.0, "beta": 0.0,  "gamma": 5.0,
                     "desc": "No Sparsity (beta=0)"},
}

# Eval config
STABILITY_SAMPLES = 20
STABILITY_EPSILON = 0.05

with open(os.path.join(DATA_DIR, "config.json")) as f:
    data_config = json.load(f)
N_MELS = data_config["n_mels"]
MAX_FRAMES = data_config["max_frames"]


# ============================================================
# MODELS (must match Programs 2 & 3)
# ============================================================

class DetectorA(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.dec1 = nn.Conv2d(16, 1, 3, padding=1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        d3 = self.dec3(e3) + e2; d2 = self.dec2(d3) + e1
        return torch.sigmoid(self.dec1(d2))


# ============================================================
# DATA LOADING
# ============================================================

def load_chunked_specs(prefix, data_dir):
    with open(os.path.join(data_dir, f"{prefix}_meta.json")) as f:
        meta = json.load(f)
    chunks = []
    for i in range(meta['n_chunks']):
        chunks.append(np.load(os.path.join(data_dir, f"{prefix}_specs_chunk{i}.npy")))
    return np.concatenate(chunks, axis=0)


def load_train_data():
    print("Loading training data...")
    train_specs = load_chunked_specs("train", DATA_DIR)
    train_labels = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))
    train_specs = (train_specs - stats['mean']) / (stats['std'] + 1e-8)
    train_specs = train_specs[:, np.newaxis, :, :]
    X = torch.tensor(train_specs, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2, pin_memory=True)
    print(f"  Train: {X.shape}")
    return loader


def load_eval_data():
    print("Loading eval data...")
    eval_specs = load_chunked_specs("eval", DATA_DIR)
    eval_labels = np.load(os.path.join(DATA_DIR, "eval_labels.npy"))
    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))
    eval_specs = (eval_specs - stats['mean']) / (stats['std'] + 1e-8)
    eval_specs = eval_specs[:, np.newaxis, :, :]
    X = torch.tensor(eval_specs, dtype=torch.float32)
    y = torch.tensor(eval_labels, dtype=torch.long)
    print(f"  Eval: {X.shape}")
    return X, y


# ============================================================
# TRAINING (same game loop as Program 3, parameterized)
# ============================================================

def adversarial_perturbation(explainer, X, epsilon):
    X_adv = X.clone().detach().requires_grad_(True)
    mask = explainer(X_adv)
    loss = mask.sum()
    grad = torch.autograd.grad(loss, X_adv, create_graph=False)[0]
    perturbation = epsilon * grad / (grad.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-8)
    return (X + perturbation).detach()


def compute_iteg_loss(detector, explainer, X, alpha, beta, gamma, epsilon):
    mask = explainer(X)

    # Faithfulness
    with torch.no_grad():
        orig_probs = torch.softmax(detector(X), dim=1)
    masked_probs = torch.softmax(detector(X * mask), dim=1)
    faithfulness_loss = F.mse_loss(masked_probs, orig_probs)

    # Sparsity
    sparsity_loss = torch.mean(torch.abs(mask))

    # Stability
    if gamma > 0:
        X_adv = adversarial_perturbation(explainer, X, epsilon)
        mask_adv = explainer(X_adv)
        stability_loss = F.mse_loss(mask, mask_adv)
    else:
        stability_loss = torch.tensor(0.0, device=X.device)

    total = alpha * faithfulness_loss + beta * sparsity_loss + gamma * stability_loss
    return total, faithfulness_loss.item(), sparsity_loss.item(), stability_loss.item()


def train_variant(variant_name, config, detector, train_loader):
    """Train one ablation variant."""
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]
    desc = config["desc"]

    final_path = os.path.join(MODEL_DIR, f"explainer_{variant_name}_final.pth")

    if os.path.exists(final_path):
        print(f"\n  [SKIP] {desc}: already trained.")
        return

    print(f"\n  Training: {desc}")
    print(f"    alpha={alpha}, beta={beta}, gamma={gamma}, epsilon={EPSILON}")

    explainer = Explainer().to(DEVICE)
    optimizer = optim.Adam(explainer.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    start_epoch = 0

    # Resume from checkpoint
    ckpt_path = os.path.join(MODEL_DIR, f"explainer_{variant_name}_ckpt.pth")
    if os.path.exists(ckpt_path):
        print(f"    [RESUME] Loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        explainer.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"    Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        explainer.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for X, y in train_loader:
            X = X.to(DEVICE)
            optimizer.zero_grad()
            loss, f, s, st = compute_iteg_loss(detector, explainer, X,
                                                alpha, beta, gamma, EPSILON)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - t0

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            # Mask health check
            explainer.eval()
            with torch.no_grad():
                for X_chk, _ in train_loader:
                    m = explainer(X_chk.to(DEVICE))
                    print(f"    Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                          f"L={avg_loss:.4f} | "
                          f"mask: mean={m.mean():.4f}, max={m.max():.4f} | "
                          f"{elapsed:.1f}s")
                    break
            explainer.train()

            try:
                torch.save({
                    'epoch': epoch,
                    'model_state': explainer.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }, ckpt_path)
            except Exception as e:
                print(f"    [WARN] Checkpoint failed: {e}")
        elif (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:2d}/{NUM_EPOCHS} | L={avg_loss:.4f} | {elapsed:.1f}s")

    # Save final
    torch.save(explainer.state_dict(), final_path)
    print(f"    [OK] Saved: {final_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_variant(variant_name, desc, detector, X_eval):
    """Evaluate one variant on faithfulness, sparsity, stability."""
    if variant_name == "full":
        model_path = os.path.join(MODEL_DIR, "explainer_final.pth")
    else:
        model_path = os.path.join(MODEL_DIR, f"explainer_{variant_name}_final.pth")

    explainer = Explainer().to(DEVICE)
    explainer.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                          weights_only=True))
    explainer.eval()

    BS = 64
    n_eval = X_eval.shape[0]
    faith_list = []
    sparse_list = []
    stab_list = []

    # Faithfulness and Sparsity
    for i in range(0, n_eval, BS):
        X = X_eval[i:i+BS].to(DEVICE)
        with torch.no_grad():
            mask = explainer(X)
            orig_probs = torch.softmax(detector(X), dim=1)
            masked_probs = torch.softmax(detector(X * mask), dim=1)
            faith = torch.abs(orig_probs - masked_probs).sum(dim=1).mean().item()
            sparse = mask.abs().mean().item()
        faith_list.append(faith)
        sparse_list.append(sparse)
        del X, mask

    # Stability (on subset)
    for i in range(0, min(n_eval, 500), BS):
        X = X_eval[i:i+BS].to(DEVICE)
        with torch.no_grad():
            mask_orig = explainer(X)
        total_diff = 0.0
        for _ in range(STABILITY_SAMPLES):
            noise = torch.randn_like(X) * STABILITY_EPSILON
            with torch.no_grad():
                mask_pert = explainer(X + noise)
            diff = (mask_orig - mask_pert).pow(2).sum(dim=(1,2,3)).sqrt().mean().item()
            total_diff += diff
        stab_list.append(total_diff / STABILITY_SAMPLES)
        del X, mask_orig

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    result = {
        'variant': variant_name,
        'description': desc,
        'faithfulness': float(np.mean(faith_list)),
        'sparsity': float(np.mean(sparse_list)),
        'stability': float(np.mean(stab_list)),
    }
    print(f"  {desc:30s} | F={result['faithfulness']:.6f} | "
          f"S={result['sparsity']:.4f} | St={result['stability']:.4f}")
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 3b: ABLATION STUDY  v1.0")
    print("=" * 60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load detector (frozen)
    print("\nLoading Detector A (frozen)...")
    detector = DetectorA().to(DEVICE)
    detector.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "detector_A_final.pth"),
        map_location=DEVICE, weights_only=True))
    detector.eval()
    for p in detector.parameters():
        p.requires_grad = False

    # ---- PHASE 1: Train variants ----
    print("\n" + "=" * 40)
    print("PHASE 1: Training Ablation Variants")
    print("=" * 40)

    train_loader = load_train_data()

    for variant_name, config in VARIANTS.items():
        train_variant(variant_name, config, detector, train_loader)

    del train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- PHASE 2: Evaluate all variants ----
    print("\n" + "=" * 40)
    print("PHASE 2: Evaluating All Variants")
    print("=" * 40)

    results_path = os.path.join(RESULTS_DIR, "results_ablation.json")

    if os.path.exists(results_path):
        print(f"\n  [SKIP] Ablation results already exist.")
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n  {'Variant':<30s} {'Faith. ↓':>10s} {'Sparse ↓':>10s} {'Stab. ↓':>10s}")
        print(f"  {'-'*60}")
        for r in results:
            print(f"  {r['description']:<30s} {r['faithfulness']:>10.6f} "
                  f"{r['sparsity']:>10.4f} {r['stability']:>10.4f}")
    else:
        X_eval, y_eval = load_eval_data()

        results = []

        # Full ITEG
        print("\nEvaluating variants...")
        results.append(evaluate_variant("full", "Full ITEG", detector, X_eval))

        # Ablation variants
        for variant_name, config in VARIANTS.items():
            results.append(evaluate_variant(variant_name, config["desc"],
                                            detector, X_eval))

        # Save
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

        # Print table
        print(f"\n  {'Variant':<30s} {'Faith. ↓':>10s} {'Sparse ↓':>10s} {'Stab. ↓':>10s}")
        print(f"  {'-'*60}")
        for r in results:
            print(f"  {r['description']:<30s} {r['faithfulness']:>10.6f} "
                  f"{r['sparsity']:>10.4f} {r['stability']:>10.4f}")

    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)
    print(f"\n>>> Run 06_figures.py to regenerate paper figures. <<<")


if __name__ == "__main__":
    main()
