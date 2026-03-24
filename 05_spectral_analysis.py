# 05_spectral_analysis.py  v1.0
"""
Program 5/6: Spectral Forensic Analysis of ITEG Masks
=======================================================
Analyzes ITEG masks to extract signal-processing-level insights:
  1. Mean mask profile per frequency band (mel bin) per attack
  2. Differential mask: spoof mask - bonafide mask (forensic signature)
  3. Cepstral analysis of masked spectrograms (MFCC characterization)
  4. Spectral centroid comparison: TTS vs VC attacks
  5. Mask energy distribution across frequency bands

These results strengthen the signal processing narrative for IEEE SPL.

Outputs saved to results/ on Drive as JSON + numpy for figure generation.

Usage in Colab:
    %run 05_spectral_analysis.py
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

with open(os.path.join(DATA_DIR, "config.json")) as f:
    data_config = json.load(f)
N_MELS = data_config["n_mels"]       # 80
MAX_FRAMES = data_config["max_frames"] # 400
SR = data_config["sr"]                # 16000

# TTS vs VC grouping
TTS_ATTACKS = {"A07", "A08", "A09", "A12"}
VC_ATTACKS = {"A13", "A15", "A17", "A19"}


# ============================================================
# MODEL ARCHITECTURES (must match Program 3)
# ============================================================

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
    """Load eval spectrograms (raw + normalized) and metadata."""
    print("Loading eval data...")

    # Raw spectrograms (for cepstral analysis)
    eval_specs_raw = load_chunked_specs("eval", DATA_DIR)

    # Normalized (for mask generation)
    stats = np.load(os.path.join(DATA_DIR, "norm_stats.npz"))
    eval_specs_norm = (eval_specs_raw - stats['mean']) / (stats['std'] + 1e-8)
    eval_specs_norm = eval_specs_norm[:, np.newaxis, :, :]

    eval_labels = np.load(os.path.join(DATA_DIR, "eval_labels.npy"))
    eval_attacks = np.load(os.path.join(DATA_DIR, "eval_attacks.npy"),
                           allow_pickle=True)

    X = torch.tensor(eval_specs_norm, dtype=torch.float32)
    print(f"  Eval: {X.shape}")
    return X, eval_specs_raw, eval_labels, eval_attacks


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_mel_to_hz(n_mels, sr, fmax=None):
    """
    Approximate Hz values for each mel bin center.
    Uses librosa's mel scale formula.
    """
    if fmax is None:
        fmax = sr / 2
    # Mel scale: m = 2595 * log10(1 + f/700)
    mel_min = 0
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    # Center of each mel band
    mel_centers_hz = (hz_points[:-2] + hz_points[1:-1]) / 2.0
    return mel_centers_hz


def frequency_band_profile(masks_np, labels, attacks, attack_list):
    """
    Compute mean mask value per mel frequency bin, grouped by attack.
    Returns dict: attack_id -> (80,) array of mean mask values.
    """
    profiles = {}

    # Bonafide profile
    bonafide_idx = labels == 0
    if bonafide_idx.sum() > 0:
        bonafide_masks = masks_np[bonafide_idx]  # (N, 80, 400)
        profiles['bonafide'] = bonafide_masks.mean(axis=(0, 2))  # (80,)

    # Per-attack profiles
    for attack_id in attack_list:
        attack_idx = attacks == attack_id
        if attack_idx.sum() > 0:
            attack_masks = masks_np[attack_idx]
            profiles[attack_id] = attack_masks.mean(axis=(0, 2))

    return profiles


def differential_mask(profiles):
    """
    Compute differential mask: mean_spoof_profile - bonafide_profile.
    Positive values = regions more important for spoof detection.
    """
    bonafide = profiles.get('bonafide', None)
    if bonafide is None:
        return {}

    diffs = {}
    for attack_id, profile in profiles.items():
        if attack_id == 'bonafide':
            continue
        diffs[attack_id] = (profile - bonafide).tolist()

    # Aggregate: TTS vs VC
    tts_profiles = [profiles[a] for a in TTS_ATTACKS if a in profiles]
    vc_profiles = [profiles[a] for a in VC_ATTACKS if a in profiles]

    if tts_profiles:
        tts_mean = np.mean(tts_profiles, axis=0)
        diffs['TTS_mean'] = (tts_mean - bonafide).tolist()
    if vc_profiles:
        vc_mean = np.mean(vc_profiles, axis=0)
        diffs['VC_mean'] = (vc_mean - bonafide).tolist()

    return diffs


def cepstral_analysis(eval_specs_raw, masks_np, labels, attacks):
    """
    Compute mean cepstral coefficients (from masked spectrograms)
    per attack type. Uses DCT on log-mel (approximates MFCCs).
    """
    from scipy.fft import dct

    n_ceps = 13  # First 13 cepstral coefficients
    results = {}

    for group_name, group_ids in [('bonafide', ['-']),
                                   ('TTS', list(TTS_ATTACKS)),
                                   ('VC', list(VC_ATTACKS))]:
        # Collect indices
        if group_name == 'bonafide':
            idx = labels == 0
        else:
            idx = np.isin(attacks, group_ids) & (labels == 1)

        if idx.sum() == 0:
            continue

        # Masked spectrograms: raw_spec * mask
        raw = eval_specs_raw[idx]        # (N, 80, 400)
        mask = masks_np[idx]             # (N, 80, 400)
        masked = raw * mask

        # Compute cepstral coefficients via DCT on log-mel
        # Average over time first -> (N, 80)
        mean_spectrum = masked.mean(axis=2)

        # DCT along frequency axis -> cepstral coefficients
        ceps = dct(mean_spectrum, type=2, axis=1, norm='ortho')[:, :n_ceps]

        # Average over samples
        mean_ceps = ceps.mean(axis=0)
        std_ceps = ceps.std(axis=0)

        results[group_name] = {
            'mean_ceps': mean_ceps.tolist(),
            'std_ceps': std_ceps.tolist(),
            'n_samples': int(idx.sum())
        }
        print(f"  {group_name:10s} (n={idx.sum():4d}): "
              f"c0={mean_ceps[0]:.2f}, c1={mean_ceps[1]:.2f}, "
              f"c2={mean_ceps[2]:.2f}, c3={mean_ceps[3]:.2f}")

    return results


def mask_energy_bands(profiles, mel_hz):
    """
    Compute mask energy in standard frequency bands:
    Low (0-500Hz), Mid (500-2000Hz), High (2000-4000Hz), Very High (4000-8000Hz)
    """
    bands = {
        'low_0_500': (0, 500),
        'mid_500_2k': (500, 2000),
        'high_2k_4k': (2000, 4000),
        'vhigh_4k_8k': (4000, 8000),
    }

    results = {}
    for attack_id, profile in profiles.items():
        band_energies = {}
        for band_name, (f_low, f_high) in bands.items():
            bin_mask = (mel_hz >= f_low) & (mel_hz < f_high)
            if bin_mask.sum() > 0:
                band_energies[band_name] = float(np.mean(profile[bin_mask]))
            else:
                band_energies[band_name] = 0.0
        results[attack_id] = band_energies

    return results


def spectral_centroid_analysis(masks_np, labels, attacks):
    """
    Compute spectral centroid and bandwidth of masks per attack.
    Centroid = weighted mean frequency. Bandwidth = weighted std.
    """
    freq_bins = np.arange(N_MELS, dtype=np.float32)
    results = {}

    for attack_id in sorted(set(attacks)):
        idx = attacks == attack_id
        if idx.sum() == 0:
            continue

        mask_subset = masks_np[idx]  # (N, 80, 400)
        # Average over time -> (N, 80)
        mask_freq = mask_subset.mean(axis=2)

        # Centroid per sample
        weights = mask_freq / (mask_freq.sum(axis=1, keepdims=True) + 1e-8)
        centroids = (weights * freq_bins[np.newaxis, :]).sum(axis=1)
        # Bandwidth per sample
        bandwidths = np.sqrt(
            (weights * (freq_bins[np.newaxis, :] - centroids[:, np.newaxis])**2).sum(axis=1))

        label = 'spoof' if labels[idx][0] == 1 else 'bonafide'
        attack_type = 'TTS' if attack_id in TTS_ATTACKS else (
            'VC' if attack_id in VC_ATTACKS else 'bonafide')

        results[attack_id] = {
            'label': label,
            'type': attack_type,
            'centroid_mean': float(centroids.mean()),
            'centroid_std': float(centroids.std()),
            'bandwidth_mean': float(bandwidths.mean()),
            'bandwidth_std': float(bandwidths.std()),
            'n_samples': int(idx.sum()),
        }

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 5/6: SPECTRAL ANALYSIS  v1.0")
    print("=" * 60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load data and models
    X_eval, eval_specs_raw, eval_labels, eval_attacks = load_eval_data()

    print("Loading explainer...")
    explainer = Explainer().to(DEVICE)
    explainer.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "explainer_final.pth"),
        map_location=DEVICE, weights_only=True))
    explainer.eval()

    # Generate masks for all eval samples
    print("\nGenerating ITEG masks...")
    all_masks = []
    BS = 64
    with torch.no_grad():
        for i in range(0, X_eval.shape[0], BS):
            X_batch = X_eval[i:i+BS].to(DEVICE)
            masks = explainer(X_batch)
            all_masks.append(masks.cpu().squeeze(1).numpy())  # (N, 80, 400)
            del X_batch, masks
    masks_np = np.concatenate(all_masks, axis=0)
    print(f"  Masks: {masks_np.shape}, mean={masks_np.mean():.4f}")

    # Mel bin to Hz mapping
    mel_hz = compute_mel_to_hz(N_MELS, SR)

    all_attacks_list = sorted(set(eval_attacks))
    spoof_attacks = [a for a in all_attacks_list if a != '-']

    # ----------------------------------------------------------
    # ANALYSIS 1: Frequency band profiles
    # ----------------------------------------------------------
    print("\n[ANALYSIS 1] Frequency band profiles per attack...")
    profiles = frequency_band_profile(masks_np, eval_labels, eval_attacks,
                                       spoof_attacks)
    # Save profiles
    profiles_save = {k: v.tolist() for k, v in profiles.items()}
    with open(os.path.join(RESULTS_DIR, "mask_profiles.json"), 'w') as f:
        json.dump(profiles_save, f, indent=2)
    print(f"  Saved profiles for {len(profiles)} groups")

    # ----------------------------------------------------------
    # ANALYSIS 2: Differential masks (spoof - bonafide)
    # ----------------------------------------------------------
    print("\n[ANALYSIS 2] Differential masks (spoof - bonafide)...")
    diffs = differential_mask(profiles)
    with open(os.path.join(RESULTS_DIR, "differential_masks.json"), 'w') as f:
        json.dump(diffs, f, indent=2)

    # Print summary: which bands show most difference
    if 'TTS_mean' in diffs:
        tts_diff = np.array(diffs['TTS_mean'])
        top_bins_tts = np.argsort(np.abs(tts_diff))[-5:][::-1]
        print(f"  TTS: top differential mel bins = {top_bins_tts} "
              f"(~{mel_hz[top_bins_tts].astype(int)} Hz)")
    if 'VC_mean' in diffs:
        vc_diff = np.array(diffs['VC_mean'])
        top_bins_vc = np.argsort(np.abs(vc_diff))[-5:][::-1]
        print(f"  VC:  top differential mel bins = {top_bins_vc} "
              f"(~{mel_hz[top_bins_vc].astype(int)} Hz)")

    # ----------------------------------------------------------
    # ANALYSIS 3: Cepstral analysis
    # ----------------------------------------------------------
    print("\n[ANALYSIS 3] Cepstral analysis of masked spectrograms...")
    cepstral_results = cepstral_analysis(eval_specs_raw, masks_np,
                                          eval_labels, eval_attacks)
    with open(os.path.join(RESULTS_DIR, "cepstral_analysis.json"), 'w') as f:
        json.dump(cepstral_results, f, indent=2)

    # ----------------------------------------------------------
    # ANALYSIS 4: Mask energy in frequency bands
    # ----------------------------------------------------------
    print("\n[ANALYSIS 4] Mask energy by frequency band...")
    band_energy = mask_energy_bands(profiles, mel_hz)
    with open(os.path.join(RESULTS_DIR, "band_energy.json"), 'w') as f:
        json.dump(band_energy, f, indent=2)

    # Print band energy table
    print(f"\n  {'Attack':<10s} {'Low':>8s} {'Mid':>8s} {'High':>8s} {'VHigh':>8s}")
    print(f"  {'-'*42}")
    for attack_id in ['bonafide'] + spoof_attacks:
        if attack_id in band_energy:
            be = band_energy[attack_id]
            print(f"  {attack_id:<10s} "
                  f"{be['low_0_500']:>8.4f} "
                  f"{be['mid_500_2k']:>8.4f} "
                  f"{be['high_2k_4k']:>8.4f} "
                  f"{be['vhigh_4k_8k']:>8.4f}")

    # ----------------------------------------------------------
    # ANALYSIS 5: Spectral centroid & bandwidth
    # ----------------------------------------------------------
    print("\n[ANALYSIS 5] Spectral centroid & bandwidth per attack...")
    centroid_results = spectral_centroid_analysis(masks_np, eval_labels,
                                                   eval_attacks)
    with open(os.path.join(RESULTS_DIR, "spectral_centroids.json"), 'w') as f:
        json.dump(centroid_results, f, indent=2)

    print(f"\n  {'Attack':<8s} {'Type':<10s} {'Centroid':>10s} {'Bandwidth':>10s}")
    print(f"  {'-'*38}")
    for attack_id, r in sorted(centroid_results.items()):
        print(f"  {attack_id:<8s} {r['type']:<10s} "
              f"{r['centroid_mean']:>8.2f}±{r['centroid_std']:.2f} "
              f"{r['bandwidth_mean']:>8.2f}±{r['bandwidth_std']:.2f}")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults in: {RESULTS_DIR}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")
    print(f"\n>>> Ready for Program 6 (Figures). <<<")


if __name__ == "__main__":
    main()
