# Music Genre Classification — Comparative DL Study

A from-scratch PyTorch implementation comparing three neural network architectures for music genre classification on the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The project evaluates how architectural depth and inductive biases affect learning over time-frequency (Mel spectrogram) representations.

---

## Overview

Audio signals are converted to log-scaled Mel spectrograms using Librosa, then fed into one of three models:

| Model | Description | Approx. Params |
|---|---|---|
| **MLP** | Flat baseline; mean-pools the time axis, then 3 fully-connected layers | ~2.7M |
| **ShallowCNN** | 3 convolutional blocks (Conv → BN → ReLU → MaxPool) | ~200K |
| **DeepCNN** | Stem + 3 residual stages with skip connections, 5+ effective conv depths | ~1.5M |

All models are evaluated on held-out test data with classification accuracy, per-class F1 score, and confusion matrices.

---

## Project Structure

```
music_classification/
├── config.py            # All hyperparameters and paths — edit this first
├── dataset.py           # GTZAN loading, Mel spectrogram extraction, augmentation
├── models.py            # MLP, ShallowCNN, DeepCNN definitions
├── train.py             # Training loop, evaluation, plotting utilities
└── run_experiments.py   # Trains all three models and writes output files
```

---

## Setup

### 1. Get the dataset

Download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and unzip it so the structure looks like:

```
gtzan/
└── genres_original/
    ├── blues/        ← 100 .wav files each
    ├── classical/
    ├── country/
    ├── disco/
    ├── hiphop/
    ├── jazz/
    ├── metal/
    ├── pop/
    ├── reggae/
    └── rock/
```

### 2. Set the dataset path

Open `config.py` and update line 6:

```python
GTZAN_ROOT = "./gtzan/genres_original"  # change this to your actual path
```

### 3. Install dependencies

```bash
pip install torch torchaudio librosa soundfile numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

## Running

**Verify models load correctly** (no dataset needed):

```bash
python models.py
```

Expected output:
```
MLP           output: (4, 10)  params: 2,xxx,xxx
ShallowCNN    output: (4, 10)  params: xxx,xxx
DeepCNN       output: (4, 10)  params: x,xxx,xxx
```

**Run all experiments:**

```bash
python run_experiments.py
```

This trains MLP → ShallowCNN → DeepCNN sequentially. After each model trains, the best checkpoint (by validation accuracy) is loaded for final test evaluation.

---

## Output Files

After `run_experiments.py` completes, the following files are written to the working directory:

| File | Contents |
|---|---|
| `training_curves.png` | Loss and accuracy curves for all three models |
| `confusion_MLP.png` | Normalised confusion matrix for MLP |
| `confusion_ShallowCNN.png` | Normalised confusion matrix for ShallowCNN |
| `confusion_DeepCNN.png` | Normalised confusion matrix for DeepCNN |
| `results_summary.txt` | Test accuracy, parameter count, and training time per model |
| `{model}_best.pt` | Best checkpoint saved during training (one per model) |

---

## Configuration

All hyperparameters live in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_RATE` | 22050 | Hz; all audio is resampled to this |
| `N_MELS` | 128 | Number of Mel filter banks |
| `AUGMENT` | `True` | Enable pitch shift, time stretch, additive noise during training |
| `BATCH_SIZE` | 32 | |
| `NUM_EPOCHS` | 50 | |
| `LEARNING_RATE` | 1e-3 | Adam optimizer |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `DROPOUT_RATE` | 0.4 | Applied in classifier heads |

---

## Audio Pipeline

```
.wav file
  └─ resample to 22 050 Hz, trim/pad to 30s
       └─ [optional augmentation: pitch shift, time stretch, noise]
            └─ Mel spectrogram  (n_fft=2048, hop=512, n_mels=128)
                 └─ power_to_dB  (log scale, ref=max)
                      └─ normalize  (zero mean, unit variance)
                           └─ shape: (1, 128, ~1292)  →  model input
```

---

## Model Architectures

### MLP
Operates on a frequency-averaged summary of the spectrogram (mean over the time axis → 128-dim vector).
```
(B, 1, 128, T) → mean over T → (B, 128)
→ FC(2048) → BN → ReLU → Dropout
→ FC(1024) → BN → ReLU → Dropout
→ FC(512)  → BN → ReLU → Dropout
→ FC(10)
```

### ShallowCNN
Three convolutional blocks with progressively more channels, followed by global average pooling.
```
(B, 1, 128, T)
→ ConvBlock(1→32,   pool)   → (B,  32, 64, T/2)
→ ConvBlock(32→64,  pool)   → (B,  64, 32, T/4)
→ ConvBlock(64→128, pool)   → (B, 128, 16, T/8)
→ AdaptiveAvgPool           → (B, 128)
→ FC(256) → ReLU → Dropout → FC(10)
```

### DeepCNN
A large-kernel stem followed by three residual stages. Each `ResidualBlock` applies two conv layers with a skip connection (1×1 projection when channels change).
```
(B, 1, 128, T)
→ Stem: ConvBlock(1→32, 7×7, pool)       → (B,  32, 64, T/2)
→ Stage1: ResBlock(32→64)  + MaxPool     → (B,  64, 32, T/4)
→ Stage2: ResBlock(64→128) + MaxPool     → (B, 128, 16, T/8)
→ Stage3: ResBlock(128→256)+ MaxPool     → (B, 256,  8, T/16)
→ AdaptiveAvgPool                        → (B, 256)
→ Dropout → FC(512) → ReLU → Dropout → FC(10)
```

---

## Training Details

- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam with L2 weight decay
- **Scheduler**: ReduceLROnPlateau — halves learning rate if validation loss does not improve for 5 epochs
- **Checkpointing**: Best model by validation accuracy is saved and reloaded for final test evaluation
- **Dataset split**: 70% train / 15% val / 15% test, stratified by genre

---

## Classes

The GTZAN dataset contains 10 genres, 100 tracks each:

`blues` · `classical` · `country` · `disco` · `hiphop` · `jazz` · `metal` · `pop` · `reggae` · `rock`