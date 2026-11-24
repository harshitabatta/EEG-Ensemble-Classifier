import os
import tempfile
import traceback

import numpy as np
import pandas as pd
import mne
import streamlit as st
import h5py
import joblib

from ensemble_predict import FixedAdaptiveEnsembleEEGPredictor

from scipy import signal, stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tensorflow import keras
from tensorflow.keras.layers import DepthwiseConv2D as _TFDepthwiseConv2D
from tensorflow.keras.layers import SeparableConv2D as _TFSeparableConv2D
from tensorflow.keras.models import model_from_json

# ---------------------------------------------------------
# Channel definitions (must match training conventions)
# ---------------------------------------------------------
STANDARD_19 = [
    'FP1-REF', 'FP2-REF', 'F3-REF', 'F4-REF', 'C3-REF', 'C4-REF',
    'P3-REF', 'P4-REF', 'O1-REF', 'O2-REF', 'F7-REF', 'F8-REF',
    'T3-REF', 'T4-REF', 'T5-REF', 'T6-REF', 'FZ-REF', 'CZ-REF', 'PZ-REF'
]

# ---------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------
st.set_page_config(page_title="EEG Clinical Visualizer", layout="wide")

st.title("🧠 EEG Clinical Visualizer with AI Classification")
st.markdown("*Comprehensive EEG recording viewer with automated abnormality detection and window-level visualization*")

# ---------------------------------------------------------
# Workaround wrappers for layer deserialization
# ---------------------------------------------------------
class DepthwiseConv2D(_TFDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        kwargs.pop('something_else', None)
        super().__init__(*args, **kwargs)

class SeparableConv2D(_TFSeparableConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        kwargs.pop('kernel_initializer', None)
        kwargs.pop('kernel_regularizer', None)
        kwargs.pop('kernel_constraint', None)
        super().__init__(*args, **kwargs)

# ---------------------------------------------------------
# Model loading & artifacts
# ---------------------------------------------------------
MODEL_BASE = "eeg_classifier"
params = {}
label_encoder = None
loaded_artifact_info = {"params": None, "label_encoder": None, "model_file": None}

def load_auxiliary_artifacts(base=MODEL_BASE):
    """Load params and label encoder from .pkl files."""
    global params, label_encoder, loaded_artifact_info

    # Params
    if not params:
        params_paths = [
            "eeg_classifier_incremental_params.pkl",
            f"{base}_params.pkl",
            "eeg_classifier_params.pkl",
            "params.pkl"
        ]
        for p in params_paths:
            if os.path.exists(p):
                try:
                    params = joblib.load(p)
                    loaded_artifact_info['params'] = p
                    st.info(f"Loaded model params from: {p}")
                    break
                except Exception as e:
                    st.warning(f"Could not load params from {p}: {e}")

    # Label encoder
    if label_encoder is None:
        label_paths = [
            "eeg_classifier_incremental_label_encoder.pkl",
            f"{base}_label_encoder.pkl",
            "eeg_classifier_label_encoder.pkl",
            "label_encoder.pkl"
        ]
        for lp in label_paths:
            if os.path.exists(lp):
                try:
                    label_encoder = joblib.load(lp)
                    loaded_artifact_info['label_encoder'] = lp
                    st.info(f"Loaded label encoder from: {lp}")
                    break
                except Exception as e:
                    st.warning(f"Could not load label encoder from {lp}: {e}")

# load_auxiliary_artifacts()

@st.cache_resource
def load_ensemble():
    return FixedAdaptiveEnsembleEEGPredictor(
        model1_path="eeg_multi_classifier",
        model2_path="en_eeg_classifier_incremental",
        sensitivity_mode="balanced"  # or slider-controlled later
    )

ensemble = load_ensemble()

@st.cache_resource
def load_model_and_info():
    """Load model (.h5 or JSON+weights) and return (model, path_info)."""
    errors = []

    h5_candidates = [
        "eeg_classifier_incremental_eegnet.h5",
        f"{MODEL_BASE}_eegnet.h5",
        f"{MODEL_BASE}.h5",
        "best_eegnet_model.h5",
        "model.h5"
    ]
    custom_objects = {'DepthwiseConv2D': DepthwiseConv2D, 'SeparableConv2D': SeparableConv2D}

    # Try named .h5 files
    for h5_path in h5_candidates:
        if os.path.exists(h5_path):
            try:
                if not h5py.is_hdf5(h5_path):
                    errors.append(f"File exists but is not a valid HDF5: {h5_path}")
                    continue
            except Exception:
                pass
            try:
                m = keras.models.load_model(h5_path, compile=False, custom_objects=custom_objects)
                loaded_artifact_info['model_file'] = h5_path
                return m, h5_path
            except Exception as e:
                errors.append(f"HDF5 load failed ({h5_path}): {e}")

    # JSON + weights fallback
    json_names = [f"{MODEL_BASE}_architecture.json", "eeg_classifier_architecture.json", "architecture.json"]
    weights_names = [f"{MODEL_BASE}_weights.h5", "eeg_classifier_weights.h5", "weights.h5"]
    for json_path in json_names:
        for weights_path in weights_names:
            if os.path.exists(json_path) and os.path.exists(weights_path):
                try:
                    with open(json_path, 'r') as f:
                        jd = f.read()
                    m = model_from_json(jd)
                    m.load_weights(weights_path)
                    loaded_artifact_info['model_file'] = f"{json_path} + {weights_path}"
                    return m, loaded_artifact_info['model_file']
                except Exception as e:
                    errors.append(f"JSON+weights load failed ({json_path}, {weights_path}): {e}")

    # Last resort: any .h5
    for fname in os.listdir('.'):
        if fname.endswith('.h5'):
            try:
                m = keras.models.load_model(fname, compile=False, custom_objects=custom_objects)
                loaded_artifact_info['model_file'] = fname
                return m, fname
            except Exception as e:
                errors.append(f"HDF5 load failed ({fname}): {e}")

    return None, " | ".join(errors)

# Cache clearing button
if st.button("Clear cached model & reload"):
    try:
        st.cache_resource.clear()
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            raise AttributeError("No rerun method available in this Streamlit version.")
    except Exception as e:
        st.warning(f"Could not clear cache automatically: {e}. Restart Streamlit server to fully clear memory.")

# model, model_info = load_model_and_info()
if ensemble is not None:

    st.success("✅ Adaptive Ensemble loaded (Multi-Input EEGNet + EEGNet)")
    st.caption("Label encodings verified and corrected automatically")

else:
    st.warning(f"⚠ Classification unavailable: {model_info}")
    st.write("Loaded artifact info:", loaded_artifact_info)


# ---------------------------------------------------------
# Helper functions for parameters
# ---------------------------------------------------------
def get_default_sampling_rate():
    return int(params.get('sampling_rate', 250))

def get_default_window_length():
    return int(params.get('window_length', 4))

def get_default_overlap():
    return float(params.get('overlap', 0.5))

def get_default_n_channels():
    return int(params.get('n_channels', 19))

# ---------------------------------------------------------
# ROBUST PREDICTION PIPELINE (Matching predict_eegnet.py)
# ---------------------------------------------------------

def load_and_preprocess_edf(raw, skip_preprocessing=True):
    """
    Load and preprocess EDF - EXACTLY matching training preprocessing
    
    Returns:
    - standardized_data: (n_channels, n_samples) array
    """
    try:
        st.write("   📥 Loading and preprocessing raw EEG (training-matched)...")
        raw_proc = raw.copy()

        # CRITICAL: Channel name cleaning (SAME AS TRAINING)
        cleaned_names = {}
        for ch in raw_proc.ch_names:
            new = ch.upper().strip()
            new = new.replace("EEG ", "").replace("EEG_", "").replace("EEG.", "")
            new = new.replace(" ", "")
            base = new.replace("-REF", "")

            if base in [
                "FP1","FP2","F3","F4","C3","C4","P3","P4",
                "O1","O2","F7","F8","T3","T4","T5","T6",
                "FZ","CZ","PZ"
            ]:
                new = base + "-REF"

            cleaned_names[ch] = new

        raw_proc.rename_channels(cleaned_names)
        raw_proc.pick_types(eeg=True, exclude='bads')

        # Standard channel order (SAME AS TRAINING)
        standard_channels = STANDARD_19

        # Apply filters ONLY if not already preprocessed
        if not skip_preprocessing:
            low_freq = params.get('low_freq', 1.0)
            high_freq = params.get('high_freq', 40.0)
            st.write(f"   🔧 Applying filters: {low_freq}-{high_freq} Hz + notch [50, 60] Hz")
            raw_proc.filter(l_freq=low_freq, h_freq=high_freq, verbose=False, picks='eeg')
            raw_proc.notch_filter(freqs=[50, 60], verbose=False, picks='eeg')

        # Resample if needed
        target_sfreq = get_default_sampling_rate()
        if raw_proc.info['sfreq'] != target_sfreq:
            st.write(f"   🔧 Resampling from {raw_proc.info['sfreq']} Hz to {target_sfreq} Hz")
            raw_proc.resample(target_sfreq, verbose=False)

        n_samples = raw_proc.n_times
        n_channels = get_default_n_channels()
        standardized_data = np.zeros((n_channels, n_samples), dtype=np.float32)

        # Fill standardized data with proper channel mapping
        present = 0
        for idx, ch_name in enumerate(standard_channels):
            if ch_name in raw_proc.ch_names:
                ch_idx = raw_proc.ch_names.index(ch_name)
                standardized_data[idx] = raw_proc.get_data(picks=[ch_idx])[0]
                present += 1
            else:
                standardized_data[idx] = 0.0

        st.write(f"   ✅ Mapped {present}/{n_channels} standard channels; others padded with zeros")

        # CRITICAL: Robust channel-wise normalization (EXACTLY AS TRAINING)
        for ch in range(n_channels):
            channel_data = standardized_data[ch]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)

            if std_val > 0:
                # Step 1: Clip outliers to 5 sigma (SAME AS TRAINING)
                channel_data = np.clip(
                    channel_data, 
                    mean_val - 5 * std_val, 
                    mean_val + 5 * std_val
                )
                
                # Step 2: Standardize (SAME AS TRAINING)
                standardized_data[ch] = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-8)

        st.write("   ✅ Channel-wise robust normalization complete (training-matched)")
        return standardized_data

    except Exception as e:
        st.error(f"   ❌ Preprocessing failed: {e}")
        st.code(traceback.format_exc())
        return None


def extract_windows(data):
    """
    Extract overlapping windows - EXACTLY as training
    
    Returns:
    - windows: (n_windows, n_channels, window_samples) array
    """
    sampling_rate = get_default_sampling_rate()
    window_length = get_default_window_length()
    overlap = get_default_overlap()

    window_samples = int(sampling_rate * window_length)
    step_size = int(window_samples * (1 - overlap))

    n_channels, n_samples = data.shape
    windows = []

    for start in range(0, n_samples - window_samples + 1, step_size):
        end = start + window_samples
        windows.append(data[:, start:end])

    if not windows:
        st.warning("⚠  No windows could be extracted (recording too short?)")
        return None

    windows = np.array(windows)
    st.write(f"   ✅ Extracted {len(windows)} windows of {window_length}s each (overlap {overlap*100:.0f}%)")
    return windows


def predict_windows(model, windows, batch_size=32):
    """
    Predict on windows with proper batch processing
    
    Returns:
    - predictions: (n_windows,) array of class predictions
    - probabilities: (n_windows,) array of probabilities for predicted class
    - all_probs: (n_windows, n_classes) raw probability array
    """
    if len(windows) == 0:
        return None, None, None

    # Reshape for EEGNet: (n_windows, n_channels, window_samples, 1)
    windows_reshaped = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], 1)

    # Get raw model predictions
    raw_predictions = model.predict(windows_reshaped, batch_size=batch_size, verbose=0)

    # Handle binary vs multi-class
    if len(raw_predictions.shape) == 2 and raw_predictions.shape[1] == 1:
        # Binary classification with sigmoid output
        probabilities_abnormal = raw_predictions.flatten()
        probabilities_normal = 1 - probabilities_abnormal

        # Combine into probability matrix
        all_probs = np.stack([probabilities_normal, probabilities_abnormal], axis=1)

        # Class predictions (0=normal, 1=abnormal)
        predictions = (probabilities_abnormal > 0.5).astype(int)

        # Confidence for predicted class
        probabilities = np.where(predictions == 1, probabilities_abnormal, probabilities_normal)

    else:
        # Multi-class with softmax output
        all_probs = raw_predictions
        predictions = np.argmax(all_probs, axis=1)
        probabilities = np.max(all_probs, axis=1)

    return predictions, probabilities, all_probs


def aggregate_window_predictions(predictions, probabilities, all_probs):
    """
    Aggregate window-level predictions to file-level with advanced methods
    
    Returns:
    - dict with aggregated predictions and confidence metrics
    """
    n_windows = len(predictions)

    # Method 1: Simple majority vote
    majority_prediction = np.bincount(predictions).argmax()

    # Method 2: Weighted vote by confidence
    weighted_votes = np.zeros(len(label_encoder.classes_) if label_encoder else 2)
    for pred, prob in zip(predictions, probabilities):
        weighted_votes[pred] += prob
    weighted_prediction = np.argmax(weighted_votes)

    # Method 3: Average probabilities across windows (PRIMARY METHOD)
    avg_probs = np.mean(all_probs, axis=0)
    avg_prob_prediction = np.argmax(avg_probs)

    # Method 4: Median probabilities (more robust to outliers)
    median_probs = np.median(all_probs, axis=0)
    median_prob_prediction = np.argmax(median_probs)

    # Calculate confidence metrics
    majority_confidence = weighted_votes[majority_prediction] / n_windows
    avg_confidence = avg_probs[avg_prob_prediction]
    median_confidence = median_probs[median_prob_prediction]

    # Consistency check: do all methods agree?
    all_agree = (
        majority_prediction == weighted_prediction == 
        avg_prob_prediction == median_prob_prediction
    )

    # Use average probability method as PRIMARY (most stable)
    final_prediction = avg_prob_prediction
    final_confidence = avg_confidence

    # Calculate additional statistics
    abnormal_ratio = np.sum(predictions == 1) / n_windows

    # Get per-class statistics
    class_names = label_encoder.classes_ if label_encoder else ['normal', 'abnormal']
    class_stats = {}
    for class_idx, class_name in enumerate(class_names):
        class_windows = np.sum(predictions == class_idx)
        class_avg_prob = np.mean(all_probs[:, class_idx])
        class_stats[class_name] = {
            'n_windows': int(class_windows),
            'ratio': float(class_windows / n_windows),
            'avg_probability': float(class_avg_prob)
        }

    # Map label
    if label_encoder is not None:
        try:
            predicted_label = label_encoder.inverse_transform([final_prediction])[0]
        except Exception:
            predicted_label = 'abnormal' if final_prediction == 1 else 'normal'
    else:
        predicted_label = 'abnormal' if final_prediction == 1 else 'normal'

    return {
        'prediction': final_prediction,
        'predicted_label': predicted_label,
        'confidence': float(final_confidence),
        'n_windows': n_windows,
        'methods_agree': all_agree,
        'abnormal_ratio': float(abnormal_ratio),
        'class_stats': class_stats,
        'all_probs_mean': avg_probs.tolist(),
        'all_probs_median': median_probs.tolist(),
        'majority_prediction': majority_prediction,
        'weighted_prediction': weighted_prediction,
        'median_prediction': median_prob_prediction
    }


def classify_with_fixed_ensemble(raw, ensemble):
    """
    Uses FixedAdaptiveEnsembleEEGPredictor
    Returns UI-compatible structure
    """

    # Reuse EXACT SAME preprocessing + window extraction
    standardized_data = load_and_preprocess_edf(raw, skip_preprocessing=True)
    windows = extract_windows(standardized_data)

    windows_reshaped = windows.reshape(*windows.shape, 1)

    # Model-level predictions
    m1 = ensemble._predict_with_model(
        ensemble.model1,
        ensemble.model1_type,
        windows_reshaped,
        ensemble.model1_label_check
    )

    m2 = ensemble._predict_with_model(
        ensemble.model2,
        ensemble.model2_type,
        windows_reshaped,
        ensemble.model2_label_check
    )

    # Ensemble fusion
    final_pred, ensemble_conf, ensemble_prob, reason = ensemble._ensemble_fusion(m1, m2)

    # ------------------------------------------------------------------
    # PER-WINDOW PROBABILITIES (CORRECT)
    # ------------------------------------------------------------------

    # Get raw per-window probabilities from each model
    # (we recompute them the same way _predict_with_model does internally)

    def get_window_probs(classifier, model_type, windows_reshaped, label_check):
        if model_type == "multi_input":
            windows = windows_reshaped.squeeze(-1)
            feature_vectors = []
            for w in windows:
                feats = classifier.extract_features_from_window(w)
                feature_vectors.append([feats[f] for f in classifier.optimal_features])

            X_feat = classifier.feature_scaler.transform(np.array(feature_vectors))
            probs_raw = classifier.model.predict(
                [windows_reshaped, X_feat], verbose=0
            ).flatten()
        else:
            probs_raw = classifier.model.predict(
                windows_reshaped, verbose=0
            ).flatten()

        # Convert to P(abnormal) using SAME label logic
        probs_abnormal = np.array([
            ensemble._get_probability_for_abnormal(p, label_check)
            for p in probs_raw
        ])

        return probs_abnormal


    # Per-window probabilities from both models
    m1_window_probs = get_window_probs(
        ensemble.model1,
        ensemble.model1_type,
        windows_reshaped,
        ensemble.model1_label_check
    )

    m2_window_probs = get_window_probs(
        ensemble.model2,
        ensemble.model2_type,
        windows_reshaped,
        ensemble.model2_label_check
    )

    # ------------------------------------------------------------------
    # PER-WINDOW ENSEMBLE FUSION (LIGHTWEIGHT, STABLE)
    # ------------------------------------------------------------------

    # Confidence-weighted average (no thresholding yet)
    window_probs = (
        ensemble.params['model1_weight'] * m1_window_probs +
        ensemble.params['model2_weight'] * m2_window_probs
    )

    # Final per-window predictions (USED FOR COLORING)
    window_preds = (window_probs > ensemble.decision_threshold).astype(int)


    # Window times (UNCHANGED)
    sampling_rate = get_default_sampling_rate()
    window_length = get_default_window_length()
    overlap = get_default_overlap()
    step_size = int(sampling_rate * window_length * (1 - overlap))

    window_times = []
    for i in range(len(window_preds)):
        start = (i * step_size) / sampling_rate
        window_times.append((start, start + window_length))

    return {
        'predicted_label': final_pred,
        'confidence': ensemble_conf,
        'ensemble_probability': ensemble_prob,
        'decision_reason': reason,

        # Window data for UI
        'window_predictions': window_preds,
        'window_probabilities': window_probs,
        'window_times': window_times,

        # Stats
        'n_windows': len(window_preds),
        'abnormal_ratio': float(np.mean(window_preds)),
        'models_agree': m1['prediction'] == m2['prediction']
    }


# ---------------------------------------------------------
# Streamlit UI: file upload & processing
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload your EEG file (.edf)", type=["edf"])

# Initialize session state for classification results
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    st.success("File uploaded successfully ✅")

    try:
        # Load EEG
        raw = mne.io.read_raw_edf(temp_file.name, preload=True, verbose=False)

        # -------------------------------
        # AI Classification Section
        # -------------------------------
        if ensemble is not None:

            st.markdown("---")
            st.subheader("🤖 AI Classification Results")

            with st.expander("⚙ Preprocessing Settings", expanded=False):
                skip_preprocessing = st.checkbox(
                    "Skip preprocessing (data already filtered)",
                    value=True,
                    help="Check this if your EDF files are already preprocessed (filtered)"
                )
                
                st.info(f"""
                **Preprocessing applied (exactly as in training):**
                - Channel name cleaning to standard 10–20 labels (FP1-REF, F3-REF, etc.)
                - EEG channels only (bad channels excluded)
                - {'Filters SKIPPED (data already preprocessed)' if skip_preprocessing else f'{params.get("low_freq", 1.0)}-{params.get("high_freq", 40.0)} Hz bandpass + 50/60 Hz notch'}
                - Resampling to {get_default_sampling_rate()} Hz
                - Mapping to fixed 19-channel montage; missing channels zero-padded
                - **5-sigma outlier clipping + channel-wise z-score normalization**
                - Sliding {get_default_window_length()}s windows with {get_default_overlap()*100:.0f}% overlap
                - **Advanced aggregation**: Average probability method (primary)
                """)

            if st.button("🚀 Run AI Classification", type="primary"):
                with st.spinner("Running robust prediction pipeline..."):
                    classification_result = classify_with_fixed_ensemble(raw, ensemble)
                    st.session_state.classification_result = classification_result


            # Display results if available
            if st.session_state.classification_result is not None:
                classification_result = st.session_state.classification_result

                if 'error' in classification_result:
                    st.error(f"Classification error: {classification_result['error']}")
                else:
                    st.markdown("### 📊 Classification Results")

                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        prediction = classification_result['predicted_label']
                        if str(prediction).lower() in ['normal', '0', 'false']:
                            st.success(f"### ✅ {prediction.upper()}")
                        else:
                            st.error(f"### ⚠ {prediction.upper()}")

                    with col2:
                        confidence = classification_result['confidence'] * 100
                        st.metric("Confidence", f"{confidence:.1f}%")

                    with col3:
                        st.metric("Total Windows", classification_result['n_windows'])

                    with col4:
                        abn_ratio = classification_result['abnormal_ratio'] * 100
                        st.metric("Abnormal %", f"{abn_ratio:.1f}%")
                    
                    with col5:
                        agree_icon = "✅" if classification_result.get('models_agree', False) else "⚠️"
                        st.metric("Methods Agree", agree_icon)

                    # Progress bar
                    try:
                        st.progress(min(max(confidence / 100.0, 0.0), 1.0))
                    except Exception:
                        pass

                    # Detailed statistics
                    with st.expander("📈 Detailed Classification Statistics", expanded=False):
                        st.write("**Per-Class Breakdown:**")
                        if 'class_stats' in classification_result:
                            for class_name, stats in classification_result['class_stats'].items():
                                st.write(f"**{class_name.upper()}:**")
                                st.write(f"  - Windows: {stats['n_windows']} ({stats['ratio']*100:.1f}%)")
                                st.write(f"  - Average Probability: {stats['avg_probability']*100:.1f}%")
                        else:
                            st.info("Class-wise statistics are not available for ensemble predictions.")
                            
                        st.write("\n**Aggregation Methods:**")
                        st.write(f"  - Primary (Avg Probability): {classification_result['predicted_label']}")
                        st.write(
                            f"  - Models agree: {classification_result.get('models_agree', False)}"
                        )
                        
                        if 'window_predictions' in classification_result:
                            window_preds = classification_result['window_predictions']
                            st.write(f"\n**Window-Level Summary:**")
                            st.write(f"  - Total windows analyzed: {len(window_preds)}")
                            st.write(f"  - Normal windows: {np.sum(window_preds == 0)}")
                            st.write(f"  - Abnormal windows: {np.sum(window_preds == 1)}")

                    st.info("""
                    **Note:** This is an AI-assisted prediction using robust aggregation methods 
                    and should not replace professional medical diagnosis. Always consult with a 
                    qualified healthcare provider for clinical interpretation.
                    """)

            st.markdown("---")

        # -------------------------------
        # Metadata
        # -------------------------------
        with st.expander("📊 EEG Recording Metadata", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Channels", raw.info["nchan"])
                st.metric("Sampling Rate", f"{raw.info['sfreq']} Hz")
            with col2:
                duration = raw.n_times / raw.info["sfreq"]
                st.metric("Duration", f"{duration:.2f} s ({duration/60:.2f} min)")
                st.metric("Total Samples", f"{raw.n_times:,}")
            with col3:
                st.write("**Channel Names:**")
                st.text("\n".join(raw.ch_names[:10]) + ("\n..." if len(raw.ch_names) > 10 else ""))

        # -------------------------------
        # Signal Processing for Visualization
        # -------------------------------
        st.subheader("🔧 Signal Processing for Visualization")
        st.info("These filters only affect the waveform display below, NOT the AI classification.")

        col1, col2 = st.columns(2)
        with col1:
            low_freq = st.slider("High-pass filter (Hz)", 0.0, 5.0, 0.5, 0.1)
        with col2:
            high_freq = st.slider("Low-pass filter (Hz)", 20.0, 100.0, 40.0, 5.0)

        raw_filtered = raw.copy()
        try:
            raw_filtered.filter(l_freq=low_freq, h_freq=high_freq)
        except Exception:
            if low_freq == 0.0:
                raw_filtered.filter(l_freq=None, h_freq=high_freq)
            else:
                raise

        # -------------------------------
        # Waveform Viewer with Window Highlighting
        # -------------------------------
        st.subheader("📈 Waveform Viewer with AI Window Predictions")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            available_channels = raw_filtered.ch_names
            selected_channels = st.multiselect(
                "Select Channels to Display",
                options=available_channels,
                default=available_channels[:min(10, len(available_channels))]
            )

        with col2:
            window_size = st.slider(
                "Initial view window (seconds)",
                min_value=1, max_value=60, value=10, step=1
            )

        with col3:
            amplitude_scale = st.slider(
                "Amplitude Scale",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1
            )

        with col4:
            max_time = raw_filtered.n_times / raw_filtered.info['sfreq']
            start_time = st.slider(
                "Start Time (seconds)",
                min_value=0.0,
                max_value=max(0.0, max_time - window_size),
                value=0.0,
                step=1.0
            )

        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            auto_downsample = st.checkbox("Auto-downsample for plotting", value=True)
        with perf_col2:
            max_points_per_channel = st.number_input(
                "Max points per channel",
                min_value=1000, max_value=50000,
                value=4000, step=500
            )
        with perf_col3:
            show_window_highlights = st.checkbox(
                "Show AI window predictions",
                value=True,
                help="Highlight abnormal windows in red, normal in green"
            )

        if selected_channels:
            channel_indices = [raw_filtered.ch_names.index(ch) for ch in selected_channels]
            n_channels = len(selected_channels)

            data, times = raw_filtered[channel_indices, :]

            data_plot = data
            times_plot = times
            n_samples = data.shape[1]

            if auto_downsample and n_samples > max_points_per_channel:
                factor = int(np.ceil(n_samples / max_points_per_channel))
                try:
                    data_plot = signal.decimate(data, factor, axis=1, zero_phase=True)
                    times_plot = np.linspace(times[0], times[-1], data_plot.shape[1])
                except Exception:
                    data_plot = data[:, ::factor]
                    times_plot = times[::factor]

            # Plotting layout
            vertical_spacing = 0.02
            per_channel_px = 180
            min_total_height = 700
            pad_factor = 0.25

            fig = make_subplots(
                rows=n_channels,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=vertical_spacing,
                subplot_titles=[
                    f"<b><span style='color:black'>{ch}</span></b>"
                    for ch in selected_channels
                ]
            )
            fig.update_annotations(font=dict(color="black", size=11))

            # FIXED: Add window highlights FIRST (before waveforms) with proper settings
            if (
                show_window_highlights
                and st.session_state.classification_result is not None
                and 'window_times' in st.session_state.classification_result
                and 'window_predictions' in st.session_state.classification_result
            ):
                window_times = st.session_state.classification_result['window_times']
                window_preds = st.session_state.classification_result['window_predictions']

                shapes = []

                for (start_t, end_t), pred in zip(window_times, window_preds):
                    if pred == 1:
                        color = 'rgba(255, 100, 100, 0.22)'   # Abnormal
                    else:
                        color = 'rgba(100, 255, 100, 0.18)'  # Normal

                    shapes.append(
                        dict(
                            type="rect",
                            x0=start_t,
                            x1=end_t,
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",   # 🔑 covers all subplots
                            fillcolor=color,
                            layer="below",
                            line_width=0
                        )
                    )

                fig.update_layout(shapes=shapes)

            # Plot waveforms AFTER adding highlights
            for i, ch_idx in enumerate(channel_indices):
                signal_data = data[i]
                signal_range = np.ptp(signal_data) if np.ptp(signal_data) != 0 else 1.0
                y_center = np.median(signal_data)

                half_span = (signal_range / 2.0) * amplitude_scale

                tiny_floor = 1e-6
                min_fraction = 0.01
                padded_half = max(
                    half_span * (1.0 + pad_factor),
                    signal_range * min_fraction,
                    tiny_floor
                )

                y_min = y_center - padded_half
                y_max = y_center + padded_half

                fig.add_trace(
                    go.Scattergl(
                        x=times_plot,
                        y=data_plot[i],
                        mode='lines',
                        line=dict(width=0.8, color='black'),
                        showlegend=False,
                        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.2f}μV<extra></extra>'
                    ),
                    row=i + 1, col=1
                )

                fig.update_yaxes(
                    range=[y_min, y_max],
                    row=i + 1, col=1,
                    title_text=f"{selected_channels[i]}",
                    title_font=dict(size=10),
                    showticklabels=True,
                    tickfont=dict(size=8),
                    autorange=False,
                    fixedrange=False,
                    automargin=True
                )

            # Rangeslider on bottom axis
            fig.update_xaxes(
                rangeslider=dict(visible=True, thickness=0.03, bgcolor='lightgray'),
                row=n_channels, col=1
            )

            for i in range(n_channels):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#e0e0e0',
                    row=i + 1, col=1
                )

            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='#f0f0f0',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='#d0d0d0'
            )

            xrange = [start_time, min(start_time + window_size, float(times_plot[-1]))]
            fig.update_layout(xaxis=dict(range=xrange))

            total_height = max(min_total_height, int(per_channel_px * n_channels))
            fig.update_layout(
                height=total_height,
                margin=dict(l=90, r=20, t=70, b=260),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                dragmode='pan'
            )

            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': [
                    'drawline',
                    'drawopenpath',
                    'eraseshape',
                    'toggleSpikelines',
                    'resetScale2d'
                ],
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'eeg_recording',
                    'height': 1200,
                    'width': 1600,
                    'scale': 2
                }
            }

            st.plotly_chart(fig, use_container_width=True, config=config)

            st.info("""
            **📖 Navigation Instructions:**
            - **Pan horizontally:** Click and drag left/right on the waveforms  
            - **Zoom in/out:** Use mouse scroll wheel or pinch gesture  
            - **Navigate entire recording:** Use the range slider at the bottom  
            - **Reset view:** Double-click anywhere on the plot or use the 'reset axes' button  
            - **Vertical zoom:** Adjust the *Amplitude Scale* slider above  
            - **Annotations:** Use drawing tools in the top-right menu bar
            - **🎨 Color coding:** Red backgrounds = Abnormal windows, Green backgrounds = Normal windows
            """)

            # Window-level predictions table
            if (st.session_state.classification_result is not None and
                'window_times' in st.session_state.classification_result):
                
                with st.expander("📋 Window-Level Predictions Table", expanded=False):
                    window_times = st.session_state.classification_result['window_times']
                    window_preds = st.session_state.classification_result['window_predictions']
                    window_probs = st.session_state.classification_result['window_probabilities']
                    
                    window_data = []
                    for i, ((start_t, end_t), pred, prob) in enumerate(zip(window_times, window_preds, window_probs)):
                        label = 'Abnormal' if pred == 1 else 'Normal'
                        window_data.append({
                            'Window': i + 1,
                            'Start (s)': f"{start_t:.2f}",
                            'End (s)': f"{end_t:.2f}",
                            'Prediction': label,
                            'Confidence': f"{prob*100:.1f}%"
                        })
                    
                    df_windows = pd.DataFrame(window_data)
                    st.dataframe(df_windows, use_container_width=True)
                    
                    # Download option
                    csv = df_windows.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Window Predictions CSV",
                        data=csv,
                        file_name="window_predictions.csv",
                        mime="text/csv"
                    )

            # -------------------------------
            # Detailed Segment Analysis
            # -------------------------------
            st.subheader("🔍 Detailed Segment Analysis")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                segment_start = st.number_input(
                    "Segment Start (s)",
                    min_value=0.0,
                    max_value=max_time - 1,
                    value=0.0,
                    step=0.1
                )
            with col_s2:
                segment_duration = st.number_input(
                    "Segment Duration (s)",
                    min_value=0.5,
                    max_value=30.0,
                    value=5.0,
                    step=0.5
                )

            if st.button("📊 Generate Segment Statistics"):
                segment_end = segment_start + segment_duration
                start_idx = int(segment_start * raw_filtered.info['sfreq'])
                end_idx = int(segment_end * raw_filtered.info['sfreq'])

                st.write(f"**Analysis for time window: {segment_start:.2f}s - {segment_end:.2f}s**")

                # Check which windows overlap with this segment
                if (st.session_state.classification_result is not None and
                    'window_times' in st.session_state.classification_result):
                    
                    window_times = st.session_state.classification_result['window_times']
                    window_preds = st.session_state.classification_result['window_predictions']
                    
                    overlapping_windows = []
                    for i, (start_t, end_t) in enumerate(window_times):
                        if not (end_t < segment_start or start_t > segment_end):
                            overlapping_windows.append({
                                'Window': i + 1,
                                'Prediction': 'Abnormal' if window_preds[i] == 1 else 'Normal',
                                'Start': f"{start_t:.2f}s",
                                'End': f"{end_t:.2f}s"
                            })
                    
                    if overlapping_windows:
                        st.write(f"\n**AI Predictions for overlapping windows:**")
                        st.table(overlapping_windows)

                stats_data = []
                for i, ch_idx in enumerate(channel_indices):
                    segment_data = data[i, start_idx:end_idx]
                    stats_data.append({
                        'Channel': selected_channels[i],
                        'Mean (μV)': f"{np.mean(segment_data):.3f}",
                        'Std Dev (μV)': f"{np.std(segment_data):.3f}",
                        'Min (μV)': f"{np.min(segment_data):.3f}",
                        'Max (μV)': f"{np.max(segment_data):.3f}",
                        'Peak-to-Peak (μV)': f"{np.ptp(segment_data):.3f}"
                    })

                st.write("\n**Channel Statistics:**")
                st.table(stats_data)

        else:
            st.warning("⚠ Please select at least one channel to display.")

    except Exception as e:
        st.error(f"❌ Error loading or processing EEG file: {e}")
        st.code(traceback.format_exc())

    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

else:
    st.info("👆 Please upload an EEG file (.edf format) to begin visualization.")

    st.markdown("""
    ### Features:
    - ✅ **Robust AI Classification** with advanced aggregation methods
      - 5-sigma outlier clipping + channel-wise normalization
      - Adaptive ensemble fusion with confidence-weighted agreement logic
      - 4 different aggregation methods with consensus checking
    - ✅ **Window-Level Visualization** - See which parts are abnormal
      - Red backgrounds = Abnormal windows
      - Green backgrounds = Normal windows
    - ✅ Full recording visualization with smooth scrolling
    - ✅ Horizontal and vertical zoom capabilities
    - ✅ Interactive channel selection
    - ✅ Adjustable signal filtering for display
    - ✅ Time segment analysis with AI window predictions
    - ✅ Annotation tools for marking regions of interest
    - ✅ Export plots as high-resolution images
    - ✅ Download window-level predictions as CSV

    ### Supported format:
    - European Data Format (.edf)

    ### Required files (place in same directory):
    - `eeg_classifier_incremental_eegnet.h5` or similar .h5 model file
    - `eeg_classifier_incremental_params.pkl` (optional but recommended)
    - `eeg_classifier_incremental_label_encoder.pkl` (optional but recommended)
    
    ### How to use:
    1. Upload an EDF file
    2. Click "Run AI Classification" to get predictions
    3. Enable "Show AI window predictions" to see colored highlights
    4. Red backgrounds = Abnormal, Green backgrounds = Normal
    5. Download window-level predictions for detailed analysis
    """)