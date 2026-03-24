# 06_figures.py  v2.0
"""
Program 6/6: Generate Paper Figures + Supplementary
=====================================================
Paper figures (2):
  Fig 1: Main comparison bar chart (Faithfulness + Stability)
  Fig 2: Qualitative masks (spectrogram + ITEG + baseline, TTS & VC)

Supplementary figures (saved to Drive only, not for paper):
  S1: Differential mask profiles (TTS vs VC)
  S2: Band energy comparison
  S3: Cepstral coefficients
  S4: Training convergence
  S5: Cross-detector transferability

All saved as PNG to Drive/figures/. Paper figures also displayed in Colab.

Usage in Colab:
    %run 06_figures.py
"""

import os
import sys
import json
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
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DRIVE_BASE = "/content/drive/MyDrive/ASVspoof_Project"
DATA_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/data")
MODEL_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/models")
RESULTS_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/results")
FIG_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(DATA_DIR, "config.json")) as f:
    data_config = json.load(f)
N_MELS = data_config["n_mels"]
MAX_FRAMES = data_config["max_frames"]
SR = data_config["sr"]

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

TTS_ATTACKS = {"A07", "A08", "A09", "A12"}
VC_ATTACKS = {"A13", "A15", "A17", "A19"}


# ============================================================
# MODELS
# ============================================================

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


def compute_mel_to_hz(n_mels, sr, fmax=None):
    if fmax is None: fmax = sr / 2
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(0, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    return (hz_points[:-2] + hz_points[1:-1]) / 2.0


def load_chunked_specs(prefix, data_dir):
    with open(os.path.join(data_dir, f"{prefix}_meta.json")) as f:
        meta = json.load(f)
    chunks = []
    for i in range(meta['n_chunks']):
        chunks.append(np.load(os.path.join(data_dir, f"{prefix}_specs_chunk{i}.npy")))
    return np.concatenate(chunks, axis=0)


def gradient_x_input_single(detector, X_single):
    X = X_single.clone().detach().requires_grad_(True)
    out = detector(X)
    out[:, 1].sum().backward()
    saliency = (X.grad * X).abs().squeeze()
    s_min, s_max = saliency.min(), saliency.max()
    return ((saliency - s_min) / (s_max - s_min + 1e-8)).detach().cpu().numpy()


def save_and_show(fig, filename, show=True):
    filepath = os.path.join(FIG_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {filename}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================
# PAPER FIGURE 1: Main Comparison
# ============================================================

def fig1_main_comparison():
    print("\n[PAPER FIG 1] Main comparison...")

    with open(os.path.join(RESULTS_DIR, "results_main.json")) as f:
        results = json.load(f)

    methods = ['ITEG', 'GradxInput', 'IntGrad', 'SHAP']
    labels = ['ITEG\n(Ours)', 'Grad×Input', 'Integrated\nGradients', 'Kernel\nSHAP']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    faith = [results[m]['faithfulness'] for m in methods]
    stab = [results[m]['stability'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Faithfulness
    bars1 = ax1.bar(labels, faith, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Faithfulness ↓')
    ax1.set_title('(a) Faithfulness')
    ax1.set_ylim(0, 1.2)
    for bar, val in zip(bars1, faith):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Stability
    bars2 = ax2.bar(labels, stab, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Stability ↓')
    ax2.set_title('(b) Stability')
    for bar, val in zip(bars2, stab):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save_and_show(fig, 'fig1_main_comparison.png', show=True)


# ============================================================
# PAPER FIGURE 2: Qualitative Masks
# ============================================================

def fig2_qualitative():
    print("\n[PAPER FIG 2] Qualitative masks...")

    # Load data
    eval_specs_raw = load_chunked_specs("eval", DATA_DIR)
    eval_labels = np.load(os.path.join(DATA_DIR, "eval_labels.npy"))
    eval_attacks = np.load(os.path.join(DATA_DIR, "eval_attacks.npy"), allow_pickle=True)
    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))

    # Load models
    explainer = Explainer().to(DEVICE)
    explainer.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "explainer_final.pth"),
        map_location=DEVICE, weights_only=True))
    explainer.eval()

    detector = DetectorA().to(DEVICE)
    detector.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "detector_A_final.pth"),
        map_location=DEVICE, weights_only=True))
    detector.eval()

    # Find one TTS sample (A07) and one VC sample (A17) correctly detected as spoof
    sample_configs = [('A07', 'TTS Attack (A07)'), ('A17', 'VC Attack (A17)')]
    samples = {}

    for target_attack, label in sample_configs:
        idx_candidates = np.where(eval_attacks == target_attack)[0]
        found = False
        for idx in idx_candidates:
            spec_raw = eval_specs_raw[idx]
            spec_norm = (spec_raw - stats['mean']) / (stats['std'] + 1e-8)
            X = torch.tensor(spec_norm[np.newaxis, np.newaxis, :, :],
                           dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                pred = detector(X).argmax(dim=1).item()
            if pred == 1:  # Correctly detected as spoof
                samples[label] = {'raw': spec_raw, 'tensor': X}
                found = True
                break
        if not found:
            # Fallback: use first sample of this attack
            idx = idx_candidates[0]
            spec_raw = eval_specs_raw[idx]
            spec_norm = (spec_raw - stats['mean']) / (stats['std'] + 1e-8)
            X = torch.tensor(spec_norm[np.newaxis, np.newaxis, :, :],
                           dtype=torch.float32).to(DEVICE)
            samples[label] = {'raw': spec_raw, 'tensor': X}

    mel_hz = compute_mel_to_hz(N_MELS, SR)
    ytick_pos = [0, 20, 40, 60, 79]
    ytick_labels = [f'{mel_hz[i]:.0f}' for i in ytick_pos]

    fig, axes = plt.subplots(2, 3, figsize=(9, 4.5))

    for row, (label, data) in enumerate(samples.items()):
        X = data['tensor']

        with torch.no_grad():
            iteg_mask = explainer(X).squeeze().cpu().numpy()
        grad_mask = gradient_x_input_single(detector, X)

        # Column 0: Input spectrogram
        ax = axes[row, 0]
        ax.imshow(data['raw'], aspect='auto', origin='lower', cmap='viridis')
        ax.set_yticks(ytick_pos); ax.set_yticklabels(ytick_labels)
        ax.set_ylabel(f'{label}\nFreq (Hz)')
        if row == 0: ax.set_title('Input Spectrogram')
        if row == 1: ax.set_xlabel('Time Frame')

        # Column 1: ITEG mask
        ax = axes[row, 1]
        ax.imshow(iteg_mask, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
        ax.set_yticks(ytick_pos); ax.set_yticklabels(ytick_labels)
        if row == 0: ax.set_title('ITEG Mask (Ours)')
        if row == 1: ax.set_xlabel('Time Frame')

        # Column 2: Grad×Input
        ax = axes[row, 2]
        ax.imshow(grad_mask, aspect='auto', origin='lower', cmap='hot', vmin=0, vmax=1)
        ax.set_yticks(ytick_pos); ax.set_yticklabels(ytick_labels)
        if row == 0: ax.set_title('Grad×Input (Baseline)')
        if row == 1: ax.set_xlabel('Time Frame')

    plt.tight_layout()
    save_and_show(fig, 'fig2_qualitative_masks.png', show=True)

    del explainer, detector
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ============================================================
# SUPPLEMENTARY FIGURES (saved to Drive only, not displayed)
# ============================================================

def supplementary_figures():
    print("\n[SUPPLEMENTARY] Generating additional figures...")

    mel_hz = compute_mel_to_hz(N_MELS, SR)

    # S1: Differential profiles
    try:
        with open(os.path.join(RESULTS_DIR, "differential_masks.json")) as f:
            diffs = json.load(f)
        with open(os.path.join(RESULTS_DIR, "mask_profiles.json")) as f:
            profiles = json.load(f)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax1.plot(mel_hz, profiles['bonafide'], 'k-', lw=1.5, label='Bonafide')
        tts_keys = [k for k in profiles if k in TTS_ATTACKS]
        vc_keys = [k for k in profiles if k in VC_ATTACKS]
        if tts_keys:
            ax1.plot(mel_hz, np.mean([profiles[k] for k in tts_keys], axis=0),
                    'b-', lw=1.5, label='TTS (mean)')
        if vc_keys:
            ax1.plot(mel_hz, np.mean([profiles[k] for k in vc_keys], axis=0),
                    'r-', lw=1.5, label='VC (mean)')
        ax1.set_ylabel('Mean Mask Value'); ax1.set_title('Mask Frequency Profiles')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        if 'TTS_mean' in diffs:
            ax2.plot(mel_hz, diffs['TTS_mean'], 'b-', lw=1.5, label='TTS − Bonafide')
        if 'VC_mean' in diffs:
            ax2.plot(mel_hz, diffs['VC_mean'], 'r-', lw=1.5, label='VC − Bonafide')
        ax2.axhline(y=0, color='k', ls='--', lw=0.5)
        ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Differential Mask')
        ax2.set_title('Differential Mask (Spoof − Bonafide)')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        save_and_show(fig, 'sup_differential_profiles.png', show=False)
    except Exception as e:
        print(f"  [WARN] S1 failed: {e}")

    # S2: Band energy
    try:
        with open(os.path.join(RESULTS_DIR, "band_energy.json")) as f:
            band_energy = json.load(f)
        bands = ['low_0_500', 'mid_500_2k', 'high_2k_4k', 'vhigh_4k_8k']
        band_labels = ['0–500', '500–2k', '2–4k', '4–8k']
        bonafide_vals = [band_energy['bonafide'][b] for b in bands]
        tts_a = [a for a in band_energy if a in TTS_ATTACKS]
        vc_a = [a for a in band_energy if a in VC_ATTACKS]
        tts_vals = np.mean([[band_energy[a][b] for b in bands] for a in tts_a], axis=0)
        vc_vals = np.mean([[band_energy[a][b] for b in bands] for a in vc_a], axis=0)
        x = np.arange(4); w = 0.25
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(x-w, bonafide_vals, w, label='Bonafide', color='#666')
        ax.bar(x, tts_vals, w, label='TTS', color='#2196F3')
        ax.bar(x+w, vc_vals, w, label='VC', color='#E91E63')
        ax.set_xticks(x); ax.set_xticklabels(band_labels)
        ax.set_ylabel('Mask Energy'); ax.set_xlabel('Frequency Band (Hz)')
        ax.set_title('Mask Energy by Frequency Band'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_and_show(fig, 'sup_band_energy.png', show=False)
    except Exception as e:
        print(f"  [WARN] S2 failed: {e}")

    # S3: Cepstral
    try:
        with open(os.path.join(RESULTS_DIR, "cepstral_analysis.json")) as f:
            ceps = json.load(f)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(13)
        for grp, col, mk in [('bonafide','#666','o'),('TTS','#2196F3','s'),('VC','#E91E63','^')]:
            if grp in ceps:
                ax.errorbar(x, ceps[grp]['mean_ceps'], yerr=ceps[grp]['std_ceps'],
                           label=grp, color=col, marker=mk, ms=4, capsize=2, lw=1)
        ax.set_xlabel('Cepstral Index'); ax.set_ylabel('Value')
        ax.set_title('Cepstral Analysis of Masked Spectrograms')
        ax.set_xticks(x); ax.set_xticklabels([f'c{i}' for i in range(13)], fontsize=7)
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_and_show(fig, 'sup_cepstral.png', show=False)
    except Exception as e:
        print(f"  [WARN] S3 failed: {e}")

    # S4: Convergence
    try:
        with open(os.path.join(MODEL_DIR, "iteg_loss_history.json")) as f:
            history = json.load(f)
        epochs = [h['epoch'] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.plot(epochs, [h['total'] for h in history], 'b-', lw=1.5)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Total Loss')
        ax1.set_title('Composite Payoff'); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, [h['faithfulness'] for h in history], 'r-', lw=1, label='Faithfulness')
        ax2.plot(epochs, [h['sparsity'] for h in history], 'g-', lw=1, label='Sparsity')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
        ax2.set_title('Components'); ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        save_and_show(fig, 'sup_convergence.png', show=False)
    except Exception as e:
        print(f"  [WARN] S4 failed: {e}")

    # S5: Transferability
    try:
        with open(os.path.join(RESULTS_DIR, "results_transfer.json")) as f:
            transfer = json.load(f)
        fig, ax = plt.subplots(figsize=(4, 3))
        vals = [transfer['faithfulness_detector_A'], transfer['faithfulness_detector_B']]
        bars = ax.bar(['Detector A\n(Source)', 'Detector B\n(Transfer)'],
                      vals, color=['#2196F3', '#FF9800'], edgecolor='black',
                      linewidth=0.5, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                   f'{val:.4f}', ha='center', fontsize=9)
        ax.set_ylabel('Faithfulness ↓')
        ax.set_title(f'Transferability (gap={transfer["transfer_gap"]:.4f})')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_and_show(fig, 'sup_transferability.png', show=False)
    except Exception as e:
        print(f"  [WARN] S5 failed: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 6/6: FIGURE GENERATION  v2.0")
    print("=" * 60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Paper figures (displayed in Colab)
    print("\n--- PAPER FIGURES ---")
    fig1_main_comparison()
    fig2_qualitative()

    # Supplementary (saved to Drive only)
    print("\n--- SUPPLEMENTARY FIGURES ---")
    supplementary_figures()

    # Summary
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    print(f"\nFigures in: {FIG_DIR}")
    for f in sorted(os.listdir(FIG_DIR)):
        fpath = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        tag = "PAPER" if f.startswith("fig") else "SUPP"
        print(f"  [{tag}] {f:45s} {size_kb:8.1f} KB")
    print(f"\n>>> All 6 programs complete. <<<")


if __name__ == "__main__":
    main()
