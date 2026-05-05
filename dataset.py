# dataset.py
# GTZAN dataset loader with Mel spectrogram extraction and optional augmentation.

import os
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

import config

CORRUPT_FILES = {"jazz.00054.wav"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load a wav file, resample to config.SAMPLE_RATE, trim/pad to config.DURATION."""
    target_len = config.SAMPLE_RATE * config.DURATION
    # librosa loads mono by default;
    waveform, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True, res_type="scipy")
    # Trim or zero-pad to exactly target_len samples
    if len(waveform) >= target_len:
        waveform = waveform[:target_len]
    else:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    return waveform.astype(np.float32)


def augment_waveform(waveform: np.ndarray) -> np.ndarray:
    """Apply one or more random augmentations to a waveform."""
    # Pitch shift
    if random.random() < 0.5:
        steps = random.uniform(-config.PITCH_SHIFT_STEPS, config.PITCH_SHIFT_STEPS)
        waveform = librosa.effects.pitch_shift(
            waveform, sr=config.SAMPLE_RATE, n_steps=steps
        )
    # Time stretch
    if random.random() < 0.5:
        rate = random.uniform(*config.TIME_STRETCH_RANGE)
        stretched = librosa.effects.time_stretch(waveform, rate=rate)
        # Re-trim/pad after stretch
        target_len = config.SAMPLE_RATE * config.DURATION
        if len(stretched) >= target_len:
            waveform = stretched[:target_len]
        else:
            waveform = np.pad(stretched, (0, target_len - len(stretched)))
    # Additive Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, config.NOISE_STD, waveform.shape).astype(np.float32)
        waveform = waveform + noise
    return waveform


def waveform_to_melspec(waveform: np.ndarray) -> np.ndarray:
    """Convert a waveform to a log-scaled, normalised Mel spectrogram.

    Returns shape: (1, N_MELS, time_frames)  — channel-first, ready for CNN.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    # Convert to log scale (dB), using top_db to avoid -inf
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to zero mean, unit variance
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    # Add channel dimension: (N_MELS, T) → (1, N_MELS, T)
    return log_mel[np.newaxis, :, :].astype(np.float32)


# ── Dataset class ──────────────────────────────────────────────────────────────

class GTZANDataset(Dataset):
    """PyTorch Dataset for GTZAN genre classification.

    Args:
        root:    Path to the genres_original/ directory.
        indices: List of integer indices into self.samples to use.
        augment: If True, apply random waveform augmentations before feature extraction.
    """

    def __init__(self, root: str, indices: list[int], augment: bool = False):
        self.augment = augment
        self.label_map = {genre: i for i, genre in enumerate(config.GENRES)}
        
        all_samples = []
        for genre in config.GENRES:
            genre_dir = os.path.join(root, genre)
            for fname in sorted(os.listdir(genre_dir)):
                if fname.endswith(".wav") and fname not in CORRUPT_FILES:
                    all_samples.append((os.path.join(genre_dir, fname), self.label_map[genre]))
        
        self.samples = [all_samples[i] for i in indices]
        
        # --- NEW: Pre-cache the data ---
        print(f"Pre-loading {len(self.samples)} samples into memory...")
        self.cached_data = []
        for path, label in self.samples:
            waveform = load_audio(path)
            # If NOT augmenting, we can pre-calculate the spec too
            if not self.augment:
                spec = waveform_to_melspec(waveform)
                self.cached_data.append((torch.from_numpy(spec), label))
            else:
                # Store waveform to skip disk I/O and resampling later
                self.cached_data.append((waveform, label))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        data, label = self.cached_data[idx]
        
        if self.augment:
            # Data is currently a waveform; augment and convert now
            waveform = augment_waveform(data)
            spec = waveform_to_melspec(waveform) # (1, N_MELS, T)
            return torch.from_numpy(spec), label
        
        # Data is already a spec tensor
        return data, label

    def __len__(self) -> int:
        return len(self.samples)


# ── Split helper ───────────────────────────────────────────────────────────────

def make_splits(root: str) -> tuple[list[int], list[int], list[int]]:
    """Return stratified train / val / test index lists.

    Stratification ensures each genre is proportionally represented in every split.
    """
    # Build a full sample list just to get labels for stratification
    all_labels = []
    for genre in config.GENRES:
        genre_dir = os.path.join(root, genre)
        n = len([f for f in os.listdir(genre_dir) if f.endswith(".wav") and f not in CORRUPT_FILES])
        all_labels.extend([config.GENRES.index(genre)] * n)

    indices = list(range(len(all_labels)))

    # First split off test set
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_RATIO,
        stratify=all_labels,
        random_state=config.RANDOM_SEED,
    )
    train_val_labels = [all_labels[i] for i in train_val_idx]

    # Then split train and val from the remainder
    relative_val = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        stratify=train_val_labels,
        random_state=config.RANDOM_SEED,
    )
    return train_idx, val_idx, test_idx


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(
    root: str = config.GTZAN_ROOT,
    augment: bool = config.AUGMENT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_idx, val_idx, test_idx = make_splits(root)

    train_ds = GTZANDataset(root, train_idx, augment=augment)
    val_ds   = GTZANDataset(root, val_idx,   augment=False)
    test_ds  = GTZANDataset(root, test_idx,  augment=False)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=pin)

    print(f"Dataset splits — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    return train_loader, val_loader, test_loader