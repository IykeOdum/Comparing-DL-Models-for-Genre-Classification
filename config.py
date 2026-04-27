# config.py
# Central configuration for all experiments

import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
GTZAN_ROOT = "./gtzan/genres_original"   # adjust to where you unzipped GTZAN

# ── Audio preprocessing ────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050   # resample all audio to this rate
DURATION      = 30      # seconds to use per track (GTZAN tracks are ~30s)
N_MELS        = 128     # number of Mel filter banks
N_FFT         = 2048    # FFT window size
HOP_LENGTH    = 512     # hop between STFT frames
# resulting spectrogram shape: (1, N_MELS, time_frames)
# time_frames ≈ ceil(SAMPLE_RATE * DURATION / HOP_LENGTH) ≈ 1292

# ── Dataset split ──────────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
RANDOM_SEED   = 42

# ── Data augmentation (applied only during training) ──────────────────────────
AUGMENT            = True
PITCH_SHIFT_STEPS  = 2      # semitones, chosen uniformly in [-n, n]
TIME_STRETCH_RANGE = (0.85, 1.15)
NOISE_STD          = 0.005  # std of Gaussian noise added to waveform

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4        # L2 regularization
DROPOUT_RATE  = 0.4

# ── Classes ────────────────────────────────────────────────────────────────────
GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]
NUM_CLASSES = len(GENRES)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"