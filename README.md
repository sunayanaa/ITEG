# ITEG: Game-Theoretic Robust Explanations for Audio Deepfake Detection

This repository contains the code and experimental pipeline for the paper:

**"Game-Theoretic Robust Explanations for Audio Deepfake Detection"**  
Sridharan Sankaran — Submitted to IEEE Signal Processing Letters

## Overview

Post-hoc explanation methods (SHAP, saliency maps, Integrated Gradients) for audio deepfake detectors are fragile — adversarial perturbations can shift attributions without altering predictions. ITEG formulates explanation as a two-player zero-sum game between an Explainer and an Adversary, producing masks that are faithful, sparse, and adversarially stable by design.

### Key Results

| Method | Faithfulness ↓ | Sparsity ↓ | Stability ↓ |
|--------|---------------|-----------|------------|
| **ITEG (Ours)** | **0.025** | 0.757 | **5.25** |
| Grad×Input | 1.001 | 0.055 | 12.62 |
| Integrated Gradients | 1.001 | 0.083 | 16.63 |
| KernelSHAP | 0.999 | 0.478 | 54.11 |

- **40× better faithfulness** than all baselines
- **2.4× more stable** than the best baseline
- **Cross-detector transfer gap: 0.016** (masks generalize across architectures)

## Repository Structure

```
├── 01_prepare_data.py       # Data preparation (v5.0)
├── 02_train_detectors.py    # CNN detector training (v2.0)
├── 03_train_iteg.py         # ITEG game training (v2.0)
├── 03b_ablation.py          # Ablation study (v1.0)
├── 04_baselines_and_eval.py # Baselines & evaluation (v1.0)
├── 05_spectral_analysis.py  # Spectral forensic analysis (v1.0)
├── 06_figures.py            # Figure generation (v2.0)
├── 07_snr_robustness.py     # Tests how ITEG mask faithfulness degrades under noise
└── README.md                # This file
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- librosa, numpy, matplotlib, scipy
- GPU (T4 recommended)

### Data

- **ASVspoof 2019 LA** dataset (train + eval partitions + protocols)
  - Download from: https://www.asvspoof.org/
  - Place zip files in: `Google Drive/ASVspoof_Project/archives/`

## Reproducing Experiments

All programs are designed to run in **Google Colab** with Google Drive mounted. Each program automatically mounts Drive, handles session disconnects, and resumes from checkpoints.

### Step 1: Data Preparation
```python
%run 01_prepare_data.py
```
- Unzips ASVspoof 2019 LA archives
- Extracts 80-band log-mel spectrograms (512-point FFT, 160-sample hop, 16 kHz)
- Saves verified 100MB chunks to Drive (survives Colab disconnects)
- **Output:** 10,000 train + 5,000 eval spectrograms (shape: 80×400)

### Step 2: Detector Training
```python
%run 02_train_detectors.py
```
- Trains Detector A (3-conv CNN, 84% accuracy) and Detector B (2-conv CNN, 79% accuracy)
- Class-weighted loss for imbalanced data (75% spoof / 25% bonafide)
- Checkpoints every 5 epochs to Drive
- **Output:** `detector_A_final.pth`, `detector_B_final.pth`

### Step 3: ITEG Training
```python
%run 03_train_iteg.py
```
- Trains the Explainer via adversarial game with frozen Detector A
- Hyperparameters: α=50 (faithfulness), β=0.05 (sparsity), γ=5 (stability), ε=0.05
- 60 epochs with mask health monitoring
- **Output:** `explainer_final.pth`, `iteg_loss_history.json`

### Step 3b: Ablation Study
```python
%run 03b_ablation.py
```
- Trains two ITEG variants: No Stability (γ=0) and No Sparsity (β=0)
- Evaluates all three variants on faithfulness, sparsity, stability
- **Output:** `explainer_no_stability_final.pth`, `explainer_no_sparsity_final.pth`, `results_ablation.json`

### Step 4: Evaluation
```python
%run 04_baselines_and_eval.py
```
- Generates masks for all methods (ITEG, Grad×Input, IG, KernelSHAP)
- Computes faithfulness, sparsity, stability on eval set
- Per-attack breakdown and cross-detector transferability
- **Output:** `results_main.json`, `results_per_attack.json`, `results_transfer.json`

### Step 5: Spectral Analysis
```python
%run 05_spectral_analysis.py
```
- Frequency band profiles per attack type
- Differential masks (spoof − bonafide)
- Cepstral analysis of masked spectrograms
- Spectral centroid and bandwidth analysis
- **Output:** `mask_profiles.json`, `cepstral_analysis.json`, `band_energy.json`, `spectral_centroids.json`

### Step 6: Figures
```python
%run 06_figures.py
```
- Paper figures: comparison bar chart + qualitative mask visualization
- Supplementary: differential profiles, band energy, cepstral, convergence, transferability
- **Output:** PNG files in `figures/` directory

### Step 7: additive noise
```python
%run 07_snr_robustness.py  


## ITEG Payoff Function

The core optimization is a minimax game:

$$\min_{\theta} \max_{\phi} L(E_\theta, A_\phi) = -\alpha \cdot I(m; h_D) + \beta \cdot H(m) + \gamma \cdot \mathbb{E}_\delta[\|m(X) - m(X+\delta)\|_2]$$

where:
- **Faithfulness** (α): Mutual information between mask and detector output
- **Sparsity** (β): Entropy of the mask — encourages binary decisions
- **Stability** (γ): Mask deviation under adversarial perturbation

At Nash equilibrium, the mask satisfies a local Lipschitz bound, ensuring robustness by construction.

