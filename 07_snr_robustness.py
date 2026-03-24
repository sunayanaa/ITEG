# 07_snr_robustness.py  v1.0
"""
Program 7: ITEG Mask Robustness Under Acoustic Noise (SNR Analysis)
=====================================================================
Tests how ITEG mask faithfulness degrades under additive Gaussian
noise at different SNR levels (clean, 20dB, 10dB, 5dB, 0dB).

This is a signal-processing-specific robustness test: unlike the
game's l2 adversary (which targets explanation stability), SNR
degradation simulates realistic acoustic channel conditions.

Quick experiment: ~5 min on GPU using existing models.

Usage in Colab:
    %run 07_snr_robustness.py
"""

import os
import json
import numpy as np
import gc
import time

try:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive')
        print("Google Drive mounted.")
    else:
        print("Google Drive already mounted.")
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

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

SNR_LEVELS = [None, 20, 10, 5, 0]  # None = clean
SNR_LABELS = ["Clean", "20 dB", "10 dB", "5 dB", "0 dB"]

with open(os.path.join(DATA_DIR, "config.json")) as f:
    data_config = json.load(f)
N_MELS = data_config["n_mels"]
MAX_FRAMES = data_config["max_frames"]


# ============================================================
# MODELS
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


def add_noise_at_snr(X_clean, snr_db):
    """
    Add Gaussian noise to normalized spectrograms at a given SNR.
    SNR is computed in the spectrogram domain (energy ratio).
    """
    signal_power = (X_clean ** 2).mean(dim=(1, 2, 3), keepdim=True)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(X_clean) * torch.sqrt(noise_power)
    return X_clean + noise


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 7: SNR ROBUSTNESS ANALYSIS  v1.0")
    print("=" * 60)

    results_path = os.path.join(RESULTS_DIR, "results_snr_robustness.json")

    if os.path.exists(results_path):
        print("\n  [SKIP] Results already exist. Loading...")
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n  {'SNR':>8s} {'Faith. ↓':>10s} {'Mask Δ':>10s} {'Det. Acc':>10s}")
        print(f"  {'-'*40}")
        for r in results:
            print(f"  {r['snr_label']:>8s} {r['faithfulness']:>10.4f} "
                  f"{r['mask_deviation']:>10.4f} {r['detector_accuracy']:>9.1f}%")
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load models
    print("\nLoading models...")
    detector = DetectorA().to(DEVICE)
    detector.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "detector_A_final.pth"),
        map_location=DEVICE, weights_only=True))
    detector.eval()
    for p in detector.parameters():
        p.requires_grad = False

    explainer = Explainer().to(DEVICE)
    explainer.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "explainer_final.pth"),
        map_location=DEVICE, weights_only=True))
    explainer.eval()

    # Load eval data
    print("Loading eval data...")
    eval_specs = load_chunked_specs("eval", DATA_DIR)
    eval_labels = np.load(os.path.join(DATA_DIR, "eval_labels.npy"))
    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))
    eval_specs_norm = (eval_specs - stats['mean']) / (stats['std'] + 1e-8)
    eval_specs_norm = eval_specs_norm[:, np.newaxis, :, :]
    X_clean = torch.tensor(eval_specs_norm, dtype=torch.float32)
    y = torch.tensor(eval_labels, dtype=torch.long)
    print(f"  Eval: {X_clean.shape}")

    # Generate clean masks (reference)
    print("\nGenerating clean reference masks...")
    BS = 64
    clean_masks = []
    with torch.no_grad():
        for i in range(0, len(X_clean), BS):
            batch = X_clean[i:i+BS].to(DEVICE)
            clean_masks.append(explainer(batch).cpu())
            del batch
    clean_masks = torch.cat(clean_masks, dim=0)
    print(f"  Clean masks: {clean_masks.shape}")

    # Evaluate at each SNR level
    print("\nEvaluating across SNR levels...")
    results = []

    for snr_db, snr_label in zip(SNR_LEVELS, SNR_LABELS):
        print(f"\n  --- {snr_label} ---")

        if snr_db is None:
            X_noisy = X_clean
        else:
            X_noisy = add_noise_at_snr(X_clean, snr_db)

        faith_list = []
        mask_dev_list = []
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(X_noisy), BS):
                X_batch = X_noisy[i:i+BS].to(DEVICE)
                y_batch = y[i:i+BS]
                clean_mask_batch = clean_masks[i:i+BS].to(DEVICE)

                # Generate mask on noisy input
                noisy_mask = explainer(X_batch)

                # Faithfulness: D(X_noisy) vs D(X_noisy * M_noisy)
                orig_probs = torch.softmax(detector(X_batch), dim=1)
                masked_probs = torch.softmax(detector(X_batch * noisy_mask), dim=1)
                faith = torch.abs(orig_probs - masked_probs).sum(dim=1).mean().item()
                faith_list.append(faith)

                # Mask deviation from clean mask
                dev = (clean_mask_batch - noisy_mask).pow(2).sum(dim=(1,2,3)).sqrt().mean().item()
                mask_dev_list.append(dev)

                # Detector accuracy on noisy input
                preds = detector(X_batch).argmax(dim=1).cpu()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

                del X_batch, noisy_mask, clean_mask_batch

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        result = {
            'snr_db': snr_db if snr_db is not None else 'inf',
            'snr_label': snr_label,
            'faithfulness': float(np.mean(faith_list)),
            'mask_deviation': float(np.mean(mask_dev_list)),
            'detector_accuracy': float(100.0 * correct / total),
        }
        results.append(result)
        print(f"    Faithfulness: {result['faithfulness']:.4f}")
        print(f"    Mask deviation from clean: {result['mask_deviation']:.4f}")
        print(f"    Detector accuracy: {result['detector_accuracy']:.1f}%")

    # Save
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Summary table
    print(f"\n  {'SNR':>8s} {'Faith. ↓':>10s} {'Mask Δ':>10s} {'Det. Acc':>10s}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['snr_label']:>8s} {r['faithfulness']:>10.4f} "
              f"{r['mask_deviation']:>10.4f} {r['detector_accuracy']:>9.1f}%")

    print("\n" + "=" * 60)
    print("SNR ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
