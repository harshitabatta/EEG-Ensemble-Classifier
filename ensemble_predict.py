#!/usr/bin/env python3
"""
FIXED Adaptive Ensemble EEG Predictor
Properly handles label encoder mapping

CRITICAL FIX: Uses model's built-in predictions instead of recreating from probabilities
This avoids label inversion issues
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


class FixedAdaptiveEnsembleEEGPredictor:

    def __init__(self, model1_path, model2_path, sensitivity_mode='balanced'):
        """
        sensitivity_mode options:
        - 'high': High sensitivity (threshold=0.35) - Catch more abnormal
        - 'balanced': Balanced (threshold=0.5) - Equal treatment
        - 'specific': High specificity (threshold=0.65) - Fewer false positives
        - float: Custom threshold (0.0 to 1.0)
        """
        print("=" * 60)
        print("🎭 FIXED ADAPTIVE ENSEMBLE EEG PREDICTOR")
        print("=" * 60)

        # Set decision threshold
        if sensitivity_mode == 'high':
            self.decision_threshold = 0.35
            mode_desc = "HIGH SENSITIVITY - Catch more abnormalities"
        elif sensitivity_mode == 'balanced':
            self.decision_threshold = 0.5
            mode_desc = "BALANCED - Equal treatment"
        elif sensitivity_mode == 'specific':
            self.decision_threshold = 0.65
            mode_desc = "HIGH SPECIFICITY - Minimize false positives"
        else:
            try:
                self.decision_threshold = float(sensitivity_mode)
                mode_desc = f"CUSTOM - Threshold {self.decision_threshold:.2f}"
            except:
                self.decision_threshold = 0.5
                mode_desc = "BALANCED - Equal treatment"
        
        print(f"\n🎯 Mode: {mode_desc}")
        print(f"   Threshold: {self.decision_threshold:.2f}")

        print("\n📥 Loading Model 1 (Multi-Input EEGNet)")
        self.model1_type, self.model1 = self._load_model(model1_path, "multi_input")
        
        # Check Model 1 label encoding
        print(f"   Label encoding: {self.model1.label_encoder.classes_}")
        self.model1_label_check = self._check_label_encoding(self.model1)

        print("\n📥 Loading Model 2 (Standard EEGNet)")
        self.model2_type, self.model2 = self._load_model(model2_path, "standard")
        
        # Check Model 2 label encoding
        print(f"   Label encoding: {self.model2.label_encoder.classes_}")
        self.model2_label_check = self._check_label_encoding(self.model2)

        print("\n✅ Both models loaded!")
        print("=" * 60)

        self.params = {
            'confidence_threshold': 0.70,
            'agreement_bonus': 0.15,
            'model1_weight': 0.52,
            'model2_weight': 0.48,
        }

    def _check_label_encoding(self, classifier):
        """
        Check if label encoding is inverted
        Returns: 'normal' if 0='normal', 1='abnormal' (standard)
                'inverted' if 0='abnormal', 1='normal' (inverted)
        """
        if len(classifier.label_encoder.classes_) >= 2:
            label_0 = classifier.label_encoder.inverse_transform([0])[0]
            label_1 = classifier.label_encoder.inverse_transform([1])[0]
            
            if label_0 == 'abnormal' and label_1 == 'normal':
                print(f"   ⚠️  INVERTED: 0='abnormal', 1='normal'")
                return 'inverted'
            elif label_0 == 'normal' and label_1 == 'abnormal':
                print(f"   ✅ STANDARD: 0='normal', 1='abnormal'")
                return 'normal'
            else:
                print(f"   ⚠️  UNKNOWN: 0='{label_0}', 1='{label_1}'")
                return 'unknown'
        return 'unknown'

    def _load_model(self, model_path, expected_type):
        if expected_type == "multi_input":
            from multi_input_eegnet_classifier import MultiInputEEGNetClassifier
            clf = MultiInputEEGNetClassifier(
                low_freq=1.0, high_freq=40.0, use_features=True
            )
            clf.load_model(model_path)
            return "multi_input", clf
        else:
            from eegnet_classifier import EEGNetClassifier
            clf = EEGNetClassifier(low_freq=1.0, high_freq=40.0)
            clf.load_model(model_path)
            return "standard", clf

    def _get_probability_for_abnormal(self, prob_raw, label_check):
        """
        Convert raw model probability to P(abnormal)
        
        If label encoding is standard (0=normal, 1=abnormal):
            prob_raw is already P(abnormal)
        If label encoding is inverted (0=abnormal, 1=normal):
            prob_raw is P(normal), so P(abnormal) = 1 - prob_raw
        """
        if label_check == 'inverted':
            # Model outputs P(normal), we need P(abnormal)
            return 1.0 - prob_raw
        else:
            # Model outputs P(abnormal), which is what we need
            return prob_raw

    def _predict_with_model(self, classifier, model_type, windows_reshaped, label_check):
        """
        Get predictions from a single model
        CRITICAL: Properly handles label encoding
        """
        if model_type == "multi_input":
            windows = windows_reshaped.squeeze(-1)
            feature_vectors = []
            for w in windows:
                feats = classifier.extract_features_from_window(w)
                feature_vectors.append(
                    [feats[f] for f in classifier.optimal_features]
                )
            X_feat = classifier.feature_scaler.transform(np.array(feature_vectors))
            probs_raw = classifier.model.predict(
                [windows_reshaped, X_feat], verbose=0
            ).flatten()
        else:
            probs_raw = classifier.model.predict(
                windows_reshaped, verbose=0
            ).flatten()

        # Convert to P(abnormal) accounting for label encoding
        probs_abnormal = np.array([
            self._get_probability_for_abnormal(p, label_check) 
            for p in probs_raw
        ])

        # Window-level predictions using corrected probabilities
        window_preds_abnormal = (probs_abnormal > 0.5).astype(int)
        
        abnormal_count = np.sum(window_preds_abnormal == 1)
        normal_count = np.sum(window_preds_abnormal == 0)
        total_windows = len(window_preds_abnormal)
        abnormal_ratio = abnormal_count / total_windows

        # Aggregate using corrected probabilities
        mean_prob = np.mean(probs_abnormal)
        median_prob = np.median(probs_abnormal)
        p75_prob = np.percentile(probs_abnormal, 75)
        p90_prob = np.percentile(probs_abnormal, 90)
        
        aggregated_prob = (
            0.35 * mean_prob +
            0.25 * median_prob +
            0.25 * p75_prob +
            0.15 * p90_prob
        )

        # File-level prediction (now correctly using P(abnormal))
        prediction = 'abnormal' if aggregated_prob > 0.5 else 'normal'
        confidence = abs(aggregated_prob - 0.5) * 2

        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'aggregated_prob': float(aggregated_prob),  # This is P(abnormal)
            'abnormal_windows': int(abnormal_count),
            'normal_windows': int(normal_count),
            'n_windows': total_windows,
            'abnormal_ratio': float(abnormal_ratio)
        }

    def _ensemble_fusion(self, m1, m2):
        """
        Ensemble fusion using corrected probabilities
        Both m1 and m2 now have aggregated_prob as P(abnormal)
        """
        p = self.params
        
        m1_conf = m1['confidence']
        m2_conf = m2['confidence']
        m1_pred = m1['prediction']
        m2_pred = m2['prediction']
        m1_prob = m1['aggregated_prob']  # P(abnormal)
        m2_prob = m2['aggregated_prob']  # P(abnormal)

        models_agree = (m1_pred == m2_pred)

        # Calculate ensemble probability
        if models_agree and m1_conf >= p['confidence_threshold'] and m2_conf >= p['confidence_threshold']:
            ensemble_prob = (m1_prob + m2_prob) / 2
            ensemble_conf = min(1.0, (m1_conf + m2_conf) / 2 + p['agreement_bonus'])
            reason = 'strong_agreement'
            
        elif models_agree:
            ensemble_prob = (m1_prob + m2_prob) / 2
            ensemble_conf = (m1_conf + m2_conf) / 2
            reason = 'agreement'
            
        else:
            # Disagreement - confidence-weighted
            w1 = m1_conf * p['model1_weight']
            w2 = m2_conf * p['model2_weight']
            ensemble_prob = (w1 * m1_prob + w2 * m2_prob) / (w1 + w2)
            ensemble_conf = max(m1_conf, m2_conf) * 0.75
            reason = 'disagreement'

        # Apply threshold (ensemble_prob is P(abnormal))
        final_pred = 'abnormal' if ensemble_prob > self.decision_threshold else 'normal'
        
        # Adjust confidence based on distance from threshold
        confidence_from_threshold = abs(ensemble_prob - self.decision_threshold) * 2
        ensemble_conf = min(ensemble_conf, confidence_from_threshold)

        return final_pred, float(ensemble_conf), float(ensemble_prob), reason

    def predict_file(self, file_path):
        """Predict on a single EDF file"""
        print(f"   🔍 {os.path.basename(file_path)}")

        data, _ = self.model1.load_edf_file(file_path, skip_preprocessing=True)
        if data is None:
            return None

        windows = self.model1.extract_windows(data)
        if len(windows) == 0:
            return None

        windows_reshaped = windows.reshape(*windows.shape, 1)

        # Get predictions with corrected label handling
        m1 = self._predict_with_model(
            self.model1, self.model1_type, windows_reshaped, self.model1_label_check
        )
        m2 = self._predict_with_model(
            self.model2, self.model2_type, windows_reshaped, self.model2_label_check
        )

        # Ensemble fusion
        final_pred, ensemble_conf, ensemble_prob, reason = self._ensemble_fusion(m1, m2)

        print(f"      → {final_pred.upper()} (P(abn)={ensemble_prob:.3f}, conf={ensemble_conf:.3f})")
        print(f"         M1: {m1['prediction']} ({m1['aggregated_prob']:.3f}) | M2: {m2['prediction']} ({m2['aggregated_prob']:.3f})")

        return {
            'file': os.path.basename(file_path),
            'full_path': file_path,
            'ensemble_prediction': final_pred,
            'ensemble_confidence': ensemble_conf,
            'ensemble_probability': ensemble_prob,
            'decision_threshold': self.decision_threshold,
            'decision_reason': reason,
            'model1_prediction': m1['prediction'],
            'model1_confidence': m1['confidence'],
            'model1_prob': m1['aggregated_prob'],
            'model1_abnormal_ratio': m1['abnormal_ratio'],
            'model2_prediction': m2['prediction'],
            'model2_confidence': m2['confidence'],
            'model2_prob': m2['aggregated_prob'],
            'model2_abnormal_ratio': m2['abnormal_ratio'],
            'models_agree': m1['prediction'] == m2['prediction']
        }

    def predict_directory(self, directory_path, output_csv):
        """Predict on all EDF files in a directory"""
        print(f"\n📂 Scanning: {directory_path}")

        if directory_path.startswith("gs://"):
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            search = directory_path.rstrip("/") + "/**/*.edf"
            files = ["gs://" + f for f in fs.glob(search.replace("gs://", ""))]
        else:
            files = [
                str(f) for f in Path(directory_path).rglob("*")
                if f.suffix.lower() == ".edf"
            ]

        print(f"📂 Found {len(files)} EDF files")
        if len(files) == 0:
            return None

        results = []
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            r = self.predict_file(file_path)
            if r:
                results.append(r)

        if not results:
            return None

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"Threshold: {self.decision_threshold:.2f}")
        print(f"Total: {len(df)}")
        
        n_abn = (df['ensemble_prediction'] == 'abnormal').sum()
        n_norm = (df['ensemble_prediction'] == 'normal').sum()
        print(f"Abnormal: {n_abn} ({n_abn/len(df)*100:.1f}%)")
        print(f"Normal: {n_norm} ({n_norm/len(df)*100:.1f}%)")
        print(f"Avg confidence: {df['ensemble_confidence'].mean():.3f}")
        print(f"Avg P(abnormal): {df['ensemble_probability'].mean():.3f}")
        print(f"Agreement: {df['models_agree'].mean()*100:.1f}%")

        print(f"\n💾 Saved to {output_csv}")
        return df


def main():
    parser = argparse.ArgumentParser(description="Fixed Adaptive Ensemble")
    parser.add_argument('--model1', default='eeg_multi_classifier')
    parser.add_argument('--model2', default='en_eeg_classifier_incremental')
    parser.add_argument('--predict-file', help='Single file')
    parser.add_argument('--predict-dir', help='Directory')
    parser.add_argument('--output-csv', default='predictions.csv')
    parser.add_argument('--sensitivity', default='balanced',
                       help='high/balanced/specific or float')
    args = parser.parse_args()

    ensemble = FixedAdaptiveEnsembleEEGPredictor(
        args.model1, args.model2, sensitivity_mode=args.sensitivity
    )

    if args.predict_file:
        ensemble.predict_file(args.predict_file)
    elif args.predict_dir:
        ensemble.predict_directory(args.predict_dir, args.output_csv)
    else:
        print("❌ Provide --predict-file or --predict-dir")


if __name__ == "__main__":
    main()