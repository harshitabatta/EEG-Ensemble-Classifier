# Ensemble Feature EEGNet Classifier

An adaptive ensemble EEG classifier that combines a **Multi-Input EEGNet** (with hand-crafted features) and a **Standard EEGNet** to detect abnormal EEG recordings. It ships with a **Streamlit web app** for interactive visualisation and a **CLI script** for batch prediction.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Required Model Files](#required-model-files)
  - [Running the Web App](#running-the-web-app)
  - [Running Batch Predictions from the CLI](#running-batch-predictions-from-the-cli)
- [Configuration](#configuration)
- [Output](#output)
- [Disclaimer](#disclaimer)

---

## Overview

The system classifies EEG recordings (in **EDF** format) as **normal** or **abnormal** by:

1. Preprocessing raw EEG signals (channel standardisation, bandpass filtering, z-score normalisation).
2. Extracting overlapping 4-second windows.
3. Running each window through two neural-network models in parallel.
4. Fusing the per-window predictions with a confidence-weighted ensemble strategy.
5. Aggregating window-level results into a single file-level decision.

---

## Repository Structure

```
ensemble_feature_eegnet_classifier/
│
├── app.py                            # Streamlit web application
├── ensemble_predict.py               # CLI batch/single-file predictor
├── multi_input_eegnet_classifier.py  # Multi-Input EEGNet classifier (with feature extraction)
├── eegnet_classifier.py              # Standard EEGNet classifier
│
├── eeg_multi_classifier_feature_scaler.pkl   # Scaler for Model 1 features
├── eeg_multi_classifier_label_encoder.pkl    # Label encoder for Model 1
├── eeg_multi_classifier_params.pkl           # Params for Model 1
├── eeg_multi_model_feature_scaler.pkl
├── eeg_multi_model_label_encoder.pkl
├── eeg_multi_model_params.pkl
│
├── en_eeg_classifier_incremental_eegnet.h5         # Keras model weights (Model 2)
├── en_eeg_classifier_incremental_label_encoder.pkl
├── en_eeg_classifier_incremental_params.pkl
│
└── README.md
```

---

## How It Works

### Preprocessing

- Channel names are normalised to the standard 10–20 montage (`FP1-REF`, `F3-REF`, …).
- Data is resampled to **250 Hz**.
- Optional bandpass filter: **1–40 Hz** + notch at **50/60 Hz**.
- Each channel is clipped to **±5σ** then **z-score normalised**.
- Missing channels are zero-padded to produce a fixed **19-channel** array.

### Windowing

- **4-second** sliding windows with **50% overlap** are extracted from the preprocessed signal.

### Ensemble Decision

| Component | Description |
|-----------|-------------|
| **Model 1** | Multi-Input EEGNet — combines raw EEG windows with a hand-crafted feature vector (spectral, statistical). |
| **Model 2** | Standard EEGNet — operates on raw EEG windows only. |
| **Fusion**  | Confidence-weighted average of per-window probabilities P(abnormal) from both models. |
| **Threshold** | Configurable sensitivity threshold (default: 0.5). |

---

## Getting Started

### Prerequisites

- Python **3.8 – 3.11**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/thanishkn/ensemble_feature_eegnet_classifier.git
cd ensemble_feature_eegnet_classifier

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow==2.13 \
            mne \
            numpy \
            pandas \
            scipy \
            scikit-learn \
            joblib \
            h5py \
            streamlit \
            plotly \
            gcsfs          # only needed for Google Cloud Storage support
```

> **Note:** TensorFlow 2.13 is recommended for compatibility with the saved `.h5` model weights. Newer versions may work but are untested.

### Required Model Files

The following artefact files must be present in the project root before running either the web app or the CLI:

| File | Purpose |
|------|---------|
| `eeg_multi_classifier_params.pkl` | Hyperparameters for Model 1 (Multi-Input EEGNet) |
| `eeg_multi_classifier_label_encoder.pkl` | Label encoder for Model 1 |
| `eeg_multi_classifier_feature_scaler.pkl` | Feature scaler for Model 1 |
| `en_eeg_classifier_incremental_eegnet.h5` | Keras weights for Model 2 (Standard EEGNet) |
| `en_eeg_classifier_incremental_params.pkl` | Hyperparameters for Model 2 |
| `en_eeg_classifier_incremental_label_encoder.pkl` | Label encoder for Model 2 |

These files are already included in the repository.  
If you retrain the models, save new artefacts to the same filenames.

---

### Running the Web App

```bash
streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`) in your browser.

**Workflow:**

1. Upload an EEG file in **EDF** format using the file uploader.
2. (Optional) Expand **Preprocessing Settings** to choose whether to apply filters.
3. Click **🚀 Run AI Classification** — results appear within seconds.
4. Toggle **Show AI window predictions** to colour-code normal (green) and abnormal (red) windows on the waveform plot.
5. Use the sliders to pan, zoom, and change amplitude scale.
6. Download per-window predictions as a CSV from the **Window-Level Predictions Table** expander.

---

### Running Batch Predictions from the CLI

`ensemble_predict.py` provides a command-line interface for processing one or many EDF files.

#### Single file

```bash
python ensemble_predict.py --predict-file /path/to/recording.edf
```

#### Directory of files

```bash
python ensemble_predict.py \
    --predict-dir /path/to/edf_folder \
    --output-csv results.csv
```

#### Full CLI reference

```
usage: ensemble_predict.py [-h]
                            [--model1 MODEL1]
                            [--model2 MODEL2]
                            [--predict-file PREDICT_FILE]
                            [--predict-dir PREDICT_DIR]
                            [--output-csv OUTPUT_CSV]
                            [--sensitivity {high,balanced,specific,<float>}]

arguments:
  --model1          Base path for Model 1 artefacts (default: eeg_multi_classifier)
  --model2          Base path for Model 2 artefacts (default: en_eeg_classifier_incremental)
  --predict-file    Path to a single EDF file
  --predict-dir     Path to a directory; all .edf files are processed recursively
  --output-csv      Output CSV file for batch results (default: predictions.csv)
  --sensitivity     Decision threshold mode:
                      high      → threshold 0.35 (catch more abnormalities)
                      balanced  → threshold 0.50 (default)
                      specific  → threshold 0.65 (fewer false positives)
                      <float>   → custom threshold, e.g. 0.45
```

#### Example with custom sensitivity

```bash
python ensemble_predict.py \
    --predict-dir ./test_edfs \
    --output-csv output/predictions.csv \
    --sensitivity high
```

---

## Configuration

Key hyperparameters (loaded automatically from the `.pkl` files):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 250 Hz | Target sampling rate after resampling |
| `window_length` | 4 s | Duration of each analysis window |
| `overlap` | 0.50 | Fractional overlap between consecutive windows |
| `n_channels` | 19 | Number of EEG channels (10–20 montage) |
| `low_freq` | 1.0 Hz | High-pass filter cutoff |
| `high_freq` | 40.0 Hz | Low-pass filter cutoff |

Ensemble fusion weights can be adjusted directly in `ensemble_predict.py`:

```python
self.params = {
    'confidence_threshold': 0.70,
    'agreement_bonus':      0.15,
    'model1_weight':        0.52,
    'model2_weight':        0.48,
}
```

---

## Output

### Web App

- **Classification badge** — NORMAL ✅ or ABNORMAL ⚠
- **Confidence %** — how far the ensemble probability is from the decision threshold
- **Abnormal %** — fraction of windows classified as abnormal
- **Colour-coded waveforms** — green = normal window, red = abnormal window
- **Downloadable CSV** — per-window start time, end time, prediction, confidence

### CLI (`predictions.csv`)

Each row corresponds to one EDF file:

| Column | Description |
|--------|-------------|
| `file` | Filename |
| `ensemble_prediction` | `normal` or `abnormal` |
| `ensemble_confidence` | Confidence score (0–1) |
| `ensemble_probability` | P(abnormal) from the ensemble |
| `model1_prediction` / `model2_prediction` | Individual model predictions |
| `models_agree` | Whether both models predicted the same label |

---

## Disclaimer

This tool is intended for **research and educational purposes only**.  
It is **not a certified medical device** and must not be used as a substitute for professional clinical interpretation by a qualified healthcare provider.
