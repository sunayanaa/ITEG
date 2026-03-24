# 01_prepare_data.py  v5.0
"""
Program 1/6: Data Preparation for ITEG Experiments
=====================================================
v5.0 Changes:
  - Chunks limited to ~100MB each (~800 samples per chunk)
  - After saving each chunk, VERIFIES it by reading back and
    checking shape + file size before proceeding
  - Failed verification triggers immediate re-save + re-verify
  - Small metadata files saved separately (always survive)
  - ~12 chunks for train, ~6 for eval

Recovery:
  - Verified chunks are skipped on re-run
  - Only unverified/missing chunks are re-extracted
  - Safe to re-run after any crash

Usage in Colab:
    %run 01_prepare_data.py
"""

import os
import sys
import zipfile
import numpy as np
import warnings
import json
import time
from collections import Counter

warnings.filterwarnings('ignore')

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

# ============================================================
# INSTALL DEPENDENCIES
# ============================================================
try:
    import librosa
except ImportError:
    print("Installing librosa...")
    os.system(f"{sys.executable} -m pip install -q librosa soundfile")
    import librosa

# ============================================================
# CONFIG
# ============================================================

DRIVE_BASE = "/content/drive/MyDrive/ASVspoof_Project"
TRAIN_ZIP = os.path.join(DRIVE_BASE, "archives/ASVspoof2019_LA_train.zip")
EVAL_ZIP = os.path.join(DRIVE_BASE, "archives/ASVspoof2019_LA_eval.zip")
PROTOCOL_ZIP = os.path.join(DRIVE_BASE, "archives/ASVspoof2019_LA_cm_protocols.zip")

LOCAL_BASE = "/content/asvspoof_local"
LOCAL_TRAIN = os.path.join(LOCAL_BASE, "train")
LOCAL_EVAL = os.path.join(LOCAL_BASE, "eval")
LOCAL_PROTOCOLS = os.path.join(LOCAL_BASE, "protocols")

OUTPUT_DIR = os.path.join(DRIVE_BASE, "Game_Theoretic_XAI/data")

# Mel-spectrogram parameters
SR = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 80
MAX_FRAMES = 400

# Attack subsets
EVAL_ATTACKS = {"A07", "A08", "A09", "A12", "A13", "A15", "A17", "A19"}
TRAIN_ATTACKS = {"A01", "A02", "A03", "A04", "A05", "A06"}

# Sample limits
TRAIN_PER_ATTACK_LIMIT = 1250   # 1250 x 6 = 7500 spoof
TRAIN_BONAFIDE_LIMIT = 2500     # Total train ~10000
EVAL_PER_ATTACK_LIMIT = 375     # 375 x 8 = 3000 spoof
EVAL_BONAFIDE_LIMIT = 2000      # Total eval ~5000

# Chunk config: ~800 samples = ~100MB per chunk
CHUNK_SIZE = 800
# Expected bytes per sample: 80 * 400 * 4 bytes = 128,000 bytes
BYTES_PER_SAMPLE = N_MELS * MAX_FRAMES * 4  # float32
MIN_CHUNK_BYTES = CHUNK_SIZE * BYTES_PER_SAMPLE * 0.8  # Allow 20% tolerance
MAX_VERIFY_RETRIES = 3


# ============================================================
# UTILITIES
# ============================================================

def ensure_unzipped(zip_path, dest_dir, name):
    """Ensure zip is extracted to local disk."""
    has_content = False
    if os.path.exists(dest_dir):
        for root, dirs, files in os.walk(dest_dir):
            if len(files) > 0:
                has_content = True
                break
    if has_content:
        print(f"  [SKIP] {name}: local files exist.")
        return
    print(f"  Unzipping {name}...")
    os.makedirs(dest_dir, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    print(f"  [OK] {name} unzipped in {time.time()-t0:.0f}s.")


def find_flac_dir(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if len([f for f in files if f.endswith('.flac')]) > 10:
            return root
    return None


def find_protocol_file(base_dir, partition):
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if partition in f.lower() and f.endswith('.txt') and 'cm' in f.lower():
                return os.path.join(root, f)
    return None


def parse_protocol(protocol_path, allowed_attacks=None,
                   per_attack_limit=None, bonafide_limit=None):
    entries = []
    attack_counts = Counter()
    bonafide_count = 0
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id, attack_id, label_str = parts[1], parts[3], parts[4]
            is_spoof = 1 if label_str == 'spoof' else 0
            if is_spoof:
                if allowed_attacks and attack_id not in allowed_attacks:
                    continue
                if per_attack_limit and attack_counts[attack_id] >= per_attack_limit:
                    continue
                attack_counts[attack_id] += 1
            else:
                if bonafide_limit and bonafide_count >= bonafide_limit:
                    continue
                bonafide_count += 1
            entries.append((file_id, attack_id, is_spoof))
    return entries


def extract_mel(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmax=SR // 2)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if log_mel.shape[1] < MAX_FRAMES:
            log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - log_mel.shape[1])),
                             mode='constant', constant_values=-80.0)
        else:
            log_mel = log_mel[:, :MAX_FRAMES]
        return log_mel.astype(np.float32)
    except:
        return None


def verify_chunk(filepath, expected_samples):
    """
    Verify a saved chunk by reading it back.
    Returns True if file exists, is readable, and has correct shape.
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"

    file_size = os.path.getsize(filepath)
    if file_size < 1000:  # Suspiciously small
        return False, f"File too small: {file_size} bytes"

    try:
        data = np.load(filepath)
        if data.shape[0] != expected_samples:
            return False, f"Wrong sample count: {data.shape[0]} vs expected {expected_samples}"
        if data.shape[1] != N_MELS or data.shape[2] != MAX_FRAMES:
            return False, f"Wrong shape: {data.shape}"
        # Quick data integrity check
        if np.isnan(data).any():
            return False, "Contains NaN values"
        del data
        return True, f"OK ({file_size / (1024*1024):.1f} MB)"
    except Exception as e:
        return False, f"Read error: {e}"


def save_and_verify_chunk(data, filepath, chunk_idx, max_retries=MAX_VERIFY_RETRIES):
    """
    Save a numpy array to Drive and verify it was saved correctly.
    Retries up to max_retries times on failure.
    """
    n_samples = data.shape[0]

    for attempt in range(1, max_retries + 1):
        try:
            np.save(filepath, data)
            # Force flush
            time.sleep(1)

            # Verify
            ok, msg = verify_chunk(filepath, n_samples)
            if ok:
                print(f"    [CHUNK {chunk_idx:2d}] Saved & verified: "
                      f"{n_samples} samples, {msg}")
                return True
            else:
                print(f"    [CHUNK {chunk_idx:2d}] Verify FAILED (attempt {attempt}): {msg}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                time.sleep(2)
        except Exception as e:
            print(f"    [CHUNK {chunk_idx:2d}] Save FAILED (attempt {attempt}): {e}")
            time.sleep(2)

    print(f"    [CHUNK {chunk_idx:2d}] FAILED after {max_retries} attempts!")
    return False


def get_chunk_path(prefix, chunk_idx):
    return os.path.join(OUTPUT_DIR, f"{prefix}_specs_chunk{chunk_idx}.npy")


def count_verified_chunks(prefix, entries_per_chunk_list):
    """Count how many consecutive verified chunks exist from the start."""
    idx = 0
    total_samples = 0
    while True:
        chunk_path = get_chunk_path(prefix, idx)
        if idx < len(entries_per_chunk_list):
            expected = entries_per_chunk_list[idx]
        else:
            # Check if file exists and is readable
            if not os.path.exists(chunk_path):
                break
            try:
                data = np.load(chunk_path)
                expected = data.shape[0]
                del data
            except:
                break

        ok, msg = verify_chunk(chunk_path, expected)
        if not ok:
            break
        total_samples += expected
        idx += 1
    return idx, total_samples


def process_partition(entries, flac_dir, partition_name, prefix):
    """
    Extract mel-spectrograms and save in verified ~100MB chunks.
    Skips already-verified chunks on resume.
    """
    total = len(entries)
    n_full_chunks = total // CHUNK_SIZE
    remainder = total % CHUNK_SIZE
    expected_chunks = n_full_chunks + (1 if remainder > 0 else 0)

    # Build list of expected samples per chunk
    entries_per_chunk = [CHUNK_SIZE] * n_full_chunks
    if remainder > 0:
        entries_per_chunk.append(remainder)

    # Check existing verified chunks
    meta_path = os.path.join(OUTPUT_DIR, f"{prefix}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('complete', False) and meta.get('n_chunks', 0) == expected_chunks:
            # Verify all chunks still exist
            all_ok = True
            for i in range(meta['n_chunks']):
                ok, _ = verify_chunk(get_chunk_path(prefix, i), entries_per_chunk[i])
                if not ok:
                    all_ok = False
                    break
            if all_ok:
                print(f"  [SKIP] {partition_name}: all {meta['n_chunks']} chunks "
                      f"verified ({meta['total_samples']} samples)")
                return

    # Find how many chunks are already done
    done_chunks, done_samples = count_verified_chunks(prefix, entries_per_chunk)
    start_entry = done_chunks * CHUNK_SIZE

    if done_chunks > 0:
        print(f"  [RESUME] {partition_name}: {done_chunks}/{expected_chunks} chunks "
              f"verified ({done_samples} samples). Resuming from entry {start_entry}.")

    specs_buffer = []
    all_labels = []
    all_attacks = []
    all_fids = []
    current_chunk = done_chunks
    failed = 0
    t0 = time.time()

    # Load existing labels if resuming
    if done_chunks > 0:
        labels_partial = os.path.join(OUTPUT_DIR, f"{prefix}_labels_building.npy")
        if os.path.exists(labels_partial):
            all_labels = list(np.load(labels_partial))
            all_attacks = list(np.load(
                os.path.join(OUTPUT_DIR, f"{prefix}_attacks_building.npy"),
                allow_pickle=True))
            all_fids = list(np.load(
                os.path.join(OUTPUT_DIR, f"{prefix}_fids_building.npy"),
                allow_pickle=True))

    for i in range(start_entry, total):
        file_id, attack_id, label = entries[i]
        audio_path = os.path.join(flac_dir, f"{file_id}.flac")

        if not os.path.exists(audio_path):
            failed += 1
            continue

        mel = extract_mel(audio_path)
        if mel is not None:
            specs_buffer.append(mel)
            all_labels.append(label)
            all_attacks.append(attack_id)
            all_fids.append(file_id)

        # Progress
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - start_entry) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{partition_name}] {i+1}/{total} "
                  f"({100*(i+1)/total:.1f}%) | "
                  f"{rate:.1f}/sec | ETA: {remaining/60:.1f} min")

        # Save chunk when buffer is full
        if len(specs_buffer) >= CHUNK_SIZE:
            chunk_data = np.array(specs_buffer[:CHUNK_SIZE])
            chunk_path = get_chunk_path(prefix, current_chunk)

            success = save_and_verify_chunk(chunk_data, chunk_path, current_chunk)
            if success:
                # Save partial labels for resume
                try:
                    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_labels_building.npy"),
                            np.array(all_labels, dtype=np.int64))
                    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_attacks_building.npy"),
                            np.array(all_attacks, dtype=object))
                    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_fids_building.npy"),
                            np.array(all_fids, dtype=object))
                except:
                    pass
                current_chunk += 1
                specs_buffer = specs_buffer[CHUNK_SIZE:]  # Keep overflow
            else:
                print(f"    [ERROR] Chunk {current_chunk} failed permanently. Stopping.")
                return

    # Save remaining samples as final chunk
    if len(specs_buffer) > 0:
        chunk_data = np.array(specs_buffer)
        chunk_path = get_chunk_path(prefix, current_chunk)
        success = save_and_verify_chunk(chunk_data, chunk_path, current_chunk)
        if success:
            current_chunk += 1

    # Save final labels, attacks, fids
    labels_arr = np.array(all_labels, dtype=np.int64)
    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_labels.npy"), labels_arr)
    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_attacks.npy"),
            np.array(all_attacks, dtype=object))
    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_fids.npy"),
            np.array(all_fids, dtype=object))

    # Save verified metadata
    meta = {
        "total_samples": len(all_labels),
        "n_chunks": current_chunk,
        "chunk_size": CHUNK_SIZE,
        "bonafide": int(np.sum(labels_arr == 0)),
        "spoof": int(np.sum(labels_arr == 1)),
        "failed": failed,
        "complete": True,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Clean up building files
    for suffix in ["_labels_building.npy", "_attacks_building.npy",
                   "_fids_building.npy"]:
        p = os.path.join(OUTPUT_DIR, f"{prefix}{suffix}")
        if os.path.exists(p):
            os.remove(p)

    elapsed = time.time() - t0
    print(f"\n  {partition_name} DONE: {meta['total_samples']} samples "
          f"in {current_chunk} verified chunks")
    print(f"    Bonafide: {meta['bonafide']}, Spoof: {meta['spoof']}")
    print(f"    Failed: {failed}, Time: {elapsed:.1f}s")


# ============================================================
# PUBLIC FUNCTION: Load chunked specs (used by Programs 2-6)
# ============================================================

def load_chunked_specs(prefix, data_dir):
    """Load and concatenate verified chunks from Drive."""
    meta_path = os.path.join(data_dir, f"{prefix}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    chunks = []
    for i in range(meta['n_chunks']):
        chunk_path = os.path.join(data_dir, f"{prefix}_specs_chunk{i}.npy")
        chunk = np.load(chunk_path)
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PROGRAM 1/6: DATA PREPARATION  v5.0")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_BASE, exist_ok=True)

    # STAGE 1-3: Unzip
    print("\n[STAGE 1] Unzipping training data...")
    ensure_unzipped(TRAIN_ZIP, LOCAL_TRAIN, "train audio")
    print("\n[STAGE 2] Unzipping eval data...")
    ensure_unzipped(EVAL_ZIP, LOCAL_EVAL, "eval audio")
    print("\n[STAGE 3] Unzipping protocol files...")
    ensure_unzipped(PROTOCOL_ZIP, LOCAL_PROTOCOLS, "protocols")

    # STAGE 4: Locate files
    print("\n[STAGE 4] Locating files...")
    train_flac_dir = find_flac_dir(LOCAL_TRAIN)
    eval_flac_dir = find_flac_dir(LOCAL_EVAL)
    if not train_flac_dir or not eval_flac_dir:
        raise FileNotFoundError("Could not find .flac directories")
    train_protocol = (find_protocol_file(LOCAL_PROTOCOLS, "train")
                      or find_protocol_file(LOCAL_TRAIN, "train"))
    eval_protocol = (find_protocol_file(LOCAL_PROTOCOLS, "eval")
                     or find_protocol_file(LOCAL_EVAL, "eval"))
    if not train_protocol or not eval_protocol:
        raise FileNotFoundError("Could not find protocol files")
    print(f"  Train: {train_flac_dir}")
    print(f"  Eval:  {eval_flac_dir}")

    # STAGE 5: Parse protocols
    print("\n[STAGE 5] Parsing protocols...")
    train_entries = parse_protocol(train_protocol, TRAIN_ATTACKS,
                                   TRAIN_PER_ATTACK_LIMIT, TRAIN_BONAFIDE_LIMIT)
    eval_entries = parse_protocol(eval_protocol, EVAL_ATTACKS,
                                  EVAL_PER_ATTACK_LIMIT, EVAL_BONAFIDE_LIMIT)

    tl = [e[2] for e in train_entries]
    tc = Counter([e[1] for e in train_entries if e[2] == 1])
    print(f"  Train: {len(train_entries)} (bonafide={tl.count(0)}, spoof={tl.count(1)})")
    print(f"    Attacks: {dict(tc)}")
    el = [e[2] for e in eval_entries]
    ec = Counter([e[1] for e in eval_entries if e[2] == 1])
    print(f"  Eval: {len(eval_entries)} (bonafide={el.count(0)}, spoof={el.count(1)})")
    print(f"    Attacks: {dict(ec)}")

    # STAGE 6-7: Extract spectrograms
    print(f"\n[STAGE 6] Train spectrograms (chunk_size={CHUNK_SIZE})...")
    process_partition(train_entries, train_flac_dir, "train", "train")

    print(f"\n[STAGE 7] Eval spectrograms (chunk_size={CHUNK_SIZE})...")
    process_partition(eval_entries, eval_flac_dir, "eval", "eval")

    # STAGE 8: Normalization stats
    print("\n[STAGE 8] Normalization statistics...")
    norm_path = os.path.join(OUTPUT_DIR, "norm_stats.npz")
    if os.path.exists(norm_path):
        stats = np.load(norm_path)
        print(f"  [SKIP] Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    else:
        train_specs = load_chunked_specs("train", OUTPUT_DIR)
        mean_val = float(train_specs.mean())
        std_val = float(train_specs.std())
        np.savez(norm_path, mean=mean_val, std=std_val)
        del train_specs
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")

    # STAGE 9: Config
    config = {
        "sr": SR, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
        "n_mels": N_MELS, "max_frames": MAX_FRAMES,
        "eval_attacks": sorted(list(EVAL_ATTACKS)),
        "train_attacks": sorted(list(TRAIN_ATTACKS)),
        "train_total": len(train_entries), "eval_total": len(eval_entries),
        "chunk_size": CHUNK_SIZE, "output_dir": OUTPUT_DIR,
    }
    with open(os.path.join(OUTPUT_DIR, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # SUMMARY
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if not f.startswith('.') and 'building' not in f:
            fpath = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  {f:45s} {size_mb:8.1f} MB")

    train_labels = np.load(os.path.join(OUTPUT_DIR, "train_labels.npy"))
    eval_labels = np.load(os.path.join(OUTPUT_DIR, "eval_labels.npy"))
    print(f"\nTrain: {len(train_labels)} "
          f"(bonafide={np.sum(train_labels==0)}, spoof={np.sum(train_labels==1)})")
    print(f"Eval:  {len(eval_labels)} "
          f"(bonafide={np.sum(eval_labels==0)}, spoof={np.sum(eval_labels==1)})")
    print(f"\n>>> Ready for Program 2 (Detector Training). <<<")


if __name__ == "__main__":
    main()
