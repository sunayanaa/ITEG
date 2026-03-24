# 04_baselines_and_eval.py  v1.0
"""
Program 4/6: Baselines & Evaluation
=====================================
Runs XAI baselines (Grad×Input, Integrated Gradients, KernelSHAP)
and evaluates all methods (including ITEG) on:
  1. Faithfulness, Sparsity, Stability (main table)
  2. Per-attack breakdown (signal processing analysis)
  3. Cross-detector transferability (ITEG masks on Detector B)

Outputs:
  - results_main.json      (Table 1 for paper)
  - results_per_attack.json (Table for SP analysis)
  - results_transfer.json  (Table 2 for paper)

Recovery:
  - Each evaluation stage saves results to Drive independently
  - Re-run skips completed stages

Usage in Colab:
    %run 04_baselines_and_eval.py
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

BATCH_SIZE = 64
STABILITY_SAMPLES = 20    # Number of perturbations for stability estimation
STABILITY_EPSILON = 0.05  # l2 perturbation bound

with open(os.path.join(DATA_DIR, "config.json")) as f:
    data_config = json.load(f)
N_MELS = data_config["n_mels"]
MAX_FRAMES = data_config["max_frames"]


# ============================================================
# MODEL ARCHITECTURES (must match Programs 2 & 3)
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


class DetectorB(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(32, 2)
    def forward(self, x):
        return self.classifier(self.features(x))


class Explainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.dec1 = nn.Conv2d(16, 1, 3, padding=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3) + e2
        d2 = self.dec2(d3) + e1
        return torch.sigmoid(self.dec1(d2))


# ============================================================
# DATA LOADING
# ============================================================

def load_chunked_specs(prefix, data_dir):
    meta_path = os.path.join(data_dir, f"{prefix}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    chunks = []
    for i in range(meta['n_chunks']):
        chunks.append(np.load(os.path.join(data_dir,
                      f"{prefix}_specs_chunk{i}.npy")))
    return np.concatenate(chunks, axis=0)


def load_eval_data():
    """Load eval spectrograms, normalize, return tensor + metadata."""
    print("Loading eval data...")
    eval_specs = load_chunked_specs("eval", DATA_DIR)
    eval_labels = np.load(os.path.join(DATA_DIR, "eval_labels.npy"))
    eval_attacks = np.load(os.path.join(DATA_DIR, "eval_attacks.npy"),
                           allow_pickle=True)

    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))
    eval_specs = (eval_specs - stats['mean']) / (stats['std'] + 1e-8)
    eval_specs = eval_specs[:, np.newaxis, :, :]

    X = torch.tensor(eval_specs, dtype=torch.float32)
    y = torch.tensor(eval_labels, dtype=torch.long)

    print(f"  Eval: {X.shape}, labels: {len(eval_labels)}, "
          f"attacks: {len(set(eval_attacks))}")
    return X, y, eval_attacks


def load_models():
    """Load Detector A, Detector B, and ITEG Explainer."""
    print("Loading models...")

    det_A = DetectorA().to(DEVICE)
    det_A.load_state_dict(torch.load(os.path.join(MODEL_DIR, "detector_A_final.pth"),
                                      map_location=DEVICE, weights_only=True))
    det_A.eval()
    for p in det_A.parameters():
        p.requires_grad = False

    det_B = DetectorB().to(DEVICE)
    det_B.load_state_dict(torch.load(os.path.join(MODEL_DIR, "detector_B_final.pth"),
                                      map_location=DEVICE, weights_only=True))
    det_B.eval()
    for p in det_B.parameters():
        p.requires_grad = False

    explainer = Explainer().to(DEVICE)
    explainer.load_state_dict(torch.load(os.path.join(MODEL_DIR, "explainer_final.pth"),
                                          map_location=DEVICE, weights_only=True))
    explainer.eval()

    print("  All models loaded.")
    return det_A, det_B, explainer


# ============================================================
# BASELINE EXPLANATION METHODS
# ============================================================

def gradient_x_input(detector, X_batch):
    """Gradient × Input saliency map, normalized to [0,1]."""
    X = X_batch.clone().detach().requires_grad_(True)
    out = detector(X)
    score = out[:, 1].sum()  # Gradient w.r.t. spoof class
    score.backward()
    saliency = (X.grad * X).abs()
    # Normalize per sample to [0,1]
    B = saliency.shape[0]
    saliency = saliency.view(B, -1)
    s_min = saliency.min(dim=1, keepdim=True)[0]
    s_max = saliency.max(dim=1, keepdim=True)[0]
    saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    return saliency.view(B, 1, N_MELS, MAX_FRAMES).detach()


def integrated_gradients(detector, X_batch, steps=30):
    """Integrated Gradients from zero baseline."""
    baseline = torch.zeros_like(X_batch).to(DEVICE)
    X = X_batch.to(DEVICE)

    # Accumulate gradients along interpolation path
    total_grads = torch.zeros_like(X)
    for k in range(1, steps + 1):
        alpha = k / steps
        interp = baseline + alpha * (X - baseline)
        interp = interp.clone().detach().requires_grad_(True)
        out = detector(interp)
        score = out[:, 1].sum()
        score.backward()
        total_grads += interp.grad

    # IG = (X - baseline) * avg_gradient
    ig = (X - baseline) * total_grads / steps
    ig = ig.abs()

    # Normalize per sample
    B = ig.shape[0]
    ig = ig.view(B, -1)
    ig_min = ig.min(dim=1, keepdim=True)[0]
    ig_max = ig.max(dim=1, keepdim=True)[0]
    ig = (ig - ig_min) / (ig_max - ig_min + 1e-8)
    return ig.view(B, 1, N_MELS, MAX_FRAMES).detach()


def kernel_shap(detector, X_batch, n_samples=50):
    """
    Simplified KernelSHAP using random binary masks.
    Approximates Shapley values by sampling coalitions.
    """
    B, C, H, W = X_batch.shape
    # Use superpixel-level masking (8x20 blocks) for efficiency
    block_h, block_w = 8, 20
    n_blocks_h = H // block_h  # 10
    n_blocks_w = W // block_w  # 20
    n_blocks = n_blocks_h * n_blocks_w  # 200

    attributions = torch.zeros(B, 1, H, W, device=DEVICE)

    with torch.no_grad():
        # Original prediction
        orig_out = torch.softmax(detector(X_batch), dim=1)[:, 1]  # P(spoof)

        for _ in range(n_samples):
            # Random binary mask over blocks
            block_mask = torch.bernoulli(
                torch.ones(B, 1, n_blocks_h, n_blocks_w, device=DEVICE) * 0.5)
            # Upsample to full resolution
            full_mask = F.interpolate(block_mask, size=(H, W), mode='nearest')

            # Evaluate masked input
            masked_out = torch.softmax(detector(X_batch * full_mask), dim=1)[:, 1]
            diff = (orig_out - masked_out).view(B, 1, 1, 1)

            # Attribution: regions in mask get credit for the difference
            attributions += diff * full_mask

    attributions = attributions.abs() / n_samples

    # Normalize per sample
    att = attributions.view(B, -1)
    a_min = att.min(dim=1, keepdim=True)[0]
    a_max = att.max(dim=1, keepdim=True)[0]
    att = (att - a_min) / (a_max - a_min + 1e-8)
    return att.view(B, 1, H, W).detach()


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_faithfulness(detector, X_batch, mask_batch):
    """
    Faithfulness = |D(X) - D(X*M)| averaged over samples.
    Lower is better — mask captures what detector needs.
    """
    with torch.no_grad():
        orig_probs = torch.softmax(detector(X_batch), dim=1)
        masked_probs = torch.softmax(detector(X_batch * mask_batch), dim=1)
        faith = torch.abs(orig_probs - masked_probs).sum(dim=1).mean()
    return faith.item()


def compute_sparsity(mask_batch):
    """
    Sparsity = mean L1 norm of mask. Lower = sparser.
    """
    return mask_batch.abs().mean().item()


def compute_stability(detector, explainer_fn, X_batch, epsilon, n_perturbations):
    """
    Stability = E[||M(X) - M(X+delta)||_2] over random perturbations.
    Lower = more stable. Only works for methods that take input and produce mask.
    """
    mask_orig = explainer_fn(X_batch)

    total_diff = 0.0
    for _ in range(n_perturbations):
        noise = torch.randn_like(X_batch) * epsilon
        X_perturbed = X_batch + noise
        mask_perturbed = explainer_fn(X_perturbed)
        diff = (mask_orig - mask_perturbed).pow(2).sum(dim=(1, 2, 3)).sqrt().mean()
        total_diff += diff.item()

    return total_diff / n_perturbations


# ============================================================
# EVALUATION PIPELINE
# ============================================================

def evaluate_all_methods(det_A, det_B, explainer, X_eval, y_eval, attacks):
    """Run full evaluation pipeline."""

    results_main_path = os.path.join(RESULTS_DIR, "results_main.json")
    results_attack_path = os.path.join(RESULTS_DIR, "results_per_attack.json")
    results_transfer_path = os.path.join(RESULTS_DIR, "results_transfer.json")

    # -------------------------------------------------------
    # STAGE 1: Generate all masks on eval set
    # -------------------------------------------------------
    print("\n[STAGE 1] Generating explanation masks...")

    n_eval = X_eval.shape[0]
    masks = {
        'ITEG': [],
        'GradxInput': [],
        'IntGrad': [],
        'SHAP': []
    }

    BS = 32  # Small batch to avoid OOM
    for i in range(0, n_eval, BS):
        X_batch = X_eval[i:i+BS].to(DEVICE)

        if (i // BS) % 10 == 0:
            print(f"  Batch {i}/{n_eval}...")

        # ITEG
        with torch.no_grad():
            masks['ITEG'].append(explainer(X_batch).cpu())

        # Grad × Input
        masks['GradxInput'].append(gradient_x_input(det_A, X_batch).cpu())

        # Integrated Gradients
        masks['IntGrad'].append(integrated_gradients(det_A, X_batch, steps=20).cpu())

        # SHAP
        masks['SHAP'].append(kernel_shap(det_A, X_batch, n_samples=30).cpu())

        # Free GPU memory
        del X_batch
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # Concatenate all masks
    for method in masks:
        masks[method] = torch.cat(masks[method], dim=0)
        print(f"  {method:12s}: shape={masks[method].shape}, "
              f"mean={masks[method].mean():.4f}, "
              f"range=[{masks[method].min():.4f}, {masks[method].max():.4f}]")

    # -------------------------------------------------------
    # STAGE 2: Compute main metrics (Table 1)
    # -------------------------------------------------------
    print("\n[STAGE 2] Computing main metrics...")

    results_main = {}
    for method_name, mask_all in masks.items():
        print(f"\n  Evaluating {method_name}...")

        faith_scores = []
        sparsity_scores = []
        stability_scores = []

        for i in range(0, n_eval, BS):
            X_batch = X_eval[i:i+BS].to(DEVICE)
            M_batch = mask_all[i:i+BS].to(DEVICE)

            # Faithfulness
            faith = compute_faithfulness(det_A, X_batch, M_batch)
            faith_scores.append(faith)

            # Sparsity
            sparsity = compute_sparsity(M_batch)
            sparsity_scores.append(sparsity)

            del X_batch, M_batch
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        # Stability (only for methods with a callable function)
        print(f"    Computing stability ({STABILITY_SAMPLES} perturbations)...")
        stab_scores = []
        for i in range(0, min(n_eval, 500), BS):  # Stability on subset (expensive)
            X_batch = X_eval[i:i+BS].to(DEVICE)

            if method_name == 'ITEG':
                fn = lambda x: explainer(x)
            elif method_name == 'GradxInput':
                fn = lambda x: gradient_x_input(det_A, x)
            elif method_name == 'IntGrad':
                fn = lambda x: integrated_gradients(det_A, x, steps=10)
            else:  # SHAP
                fn = lambda x: kernel_shap(det_A, x, n_samples=15)

            stab = compute_stability(det_A, fn, X_batch,
                                     STABILITY_EPSILON, STABILITY_SAMPLES)
            stab_scores.append(stab)

            del X_batch
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        results_main[method_name] = {
            'faithfulness': float(np.mean(faith_scores)),
            'sparsity': float(np.mean(sparsity_scores)),
            'stability': float(np.mean(stab_scores)),
        }
        r = results_main[method_name]
        print(f"    F={r['faithfulness']:.6f}, "
              f"S={r['sparsity']:.4f}, "
              f"St={r['stability']:.6f}")

    # Save
    with open(results_main_path, 'w') as f:
        json.dump(results_main, f, indent=2)
    print(f"\n  Main results saved to {results_main_path}")

    # Print table
    print("\n" + "=" * 60)
    print("TABLE 1: Comparison of Explanation Methods")
    print("=" * 60)
    print(f"  {'Method':<15s} {'Faith. ↓':>10s} {'Sparsity ↓':>12s} {'Stab. ↓':>10s}")
    print(f"  {'-'*47}")
    for method_name, r in results_main.items():
        print(f"  {method_name:<15s} {r['faithfulness']:>10.6f} "
              f"{r['sparsity']:>12.4f} {r['stability']:>10.6f}")

    # -------------------------------------------------------
    # STAGE 3: Per-attack breakdown (ITEG only)
    # -------------------------------------------------------
    print("\n[STAGE 3] Per-attack analysis (ITEG)...")

    iteg_masks = masks['ITEG']
    results_attack = {}

    for attack_id in sorted(set(attacks)):
        attack_mask = attacks == attack_id
        indices = np.where(attack_mask)[0]

        if len(indices) == 0:
            continue

        faith_list = []
        for i in range(0, len(indices), BS):
            idx_batch = indices[i:i+BS]
            X_batch = X_eval[idx_batch].to(DEVICE)
            M_batch = iteg_masks[idx_batch].to(DEVICE)
            faith = compute_faithfulness(det_A, X_batch, M_batch)
            faith_list.append(faith)
            del X_batch, M_batch

        # Mask centroid (weighted mean frequency bin)
        mask_subset = iteg_masks[indices]  # (N, 1, 80, 400)
        mask_2d = mask_subset.squeeze(1)   # (N, 80, 400)
        freq_bins = torch.arange(N_MELS, dtype=torch.float32).view(1, -1, 1)
        weighted_freq = (mask_2d * freq_bins).sum(dim=(1, 2)) / (mask_2d.sum(dim=(1, 2)) + 1e-8)
        mean_centroid = weighted_freq.mean().item()

        # Mean mask per frequency band (averaged over time)
        mean_mask_profile = mask_2d.mean(dim=(0, 2)).numpy()  # (80,)

        label = y_eval[indices[0]].item()
        results_attack[attack_id] = {
            'label': 'spoof' if label == 1 else 'bonafide',
            'n_samples': len(indices),
            'faithfulness': float(np.mean(faith_list)),
            'sparsity': float(mask_subset.abs().mean()),
            'spectral_centroid': float(mean_centroid),
            'mask_profile': mean_mask_profile.tolist(),
        }
        print(f"  {attack_id:5s} | {results_attack[attack_id]['label']:8s} | "
              f"n={len(indices):4d} | "
              f"F={results_attack[attack_id]['faithfulness']:.6f} | "
              f"centroid={mean_centroid:.1f}")

    with open(results_attack_path, 'w') as f:
        json.dump(results_attack, f, indent=2)
    print(f"  Per-attack results saved to {results_attack_path}")

    # -------------------------------------------------------
    # STAGE 4: Cross-detector transferability
    # -------------------------------------------------------
    print("\n[STAGE 4] Cross-detector transferability...")

    faith_A_list = []
    faith_B_list = []

    for i in range(0, n_eval, BS):
        X_batch = X_eval[i:i+BS].to(DEVICE)
        M_batch = iteg_masks[i:i+BS].to(DEVICE)

        faith_A = compute_faithfulness(det_A, X_batch, M_batch)
        faith_B = compute_faithfulness(det_B, X_batch, M_batch)

        faith_A_list.append(faith_A)
        faith_B_list.append(faith_B)

        del X_batch, M_batch

    results_transfer = {
        'faithfulness_detector_A': float(np.mean(faith_A_list)),
        'faithfulness_detector_B': float(np.mean(faith_B_list)),
        'transfer_gap': float(abs(np.mean(faith_A_list) - np.mean(faith_B_list))),
    }
    print(f"  Faithfulness on D_A (source):   {results_transfer['faithfulness_detector_A']:.6f}")
    print(f"  Faithfulness on D_B (transfer): {results_transfer['faithfulness_detector_B']:.6f}")
    print(f"  Transfer gap:                   {results_transfer['transfer_gap']:.6f}")

    with open(results_transfer_path, 'w') as f:
        json.dump(results_transfer, f, indent=2)
    print(f"  Transfer results saved to {results_transfer_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 4/6: BASELINES & EVALUATION  v1.0")
    print("=" * 60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    det_A, det_B, explainer = load_models()
    X_eval, y_eval, attacks = load_eval_data()

    evaluate_all_methods(det_A, det_B, explainer, X_eval, y_eval, attacks)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults in: {RESULTS_DIR}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")
    print(f"\n>>> Ready for Program 5 (Spectral Analysis). <<<")


if __name__ == "__main__":
    main()
