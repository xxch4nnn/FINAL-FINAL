"""
ML_Pipeline_Prep.py
Trains and compares SVM and Random Forest models for piano motion classification.
Includes hyperparameter tuning via RandomizedSearchCV and comprehensive evaluation.
Refactored to use shared feature_engine column definitions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
import time
import logging
import joblib
import sys

# Import shared feature engine
try:
    from feature_engine import FEATURE_COLUMNS
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from feature_engine import FEATURE_COLUMNS

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, classification_report
)
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PianoMotionMLPipeline:
    """
    Machine Learning pipeline for piano motion classification.
    Compares SVM and Random Forest models with comprehensive evaluation.
    """

    def __init__(self, features_csv: str = None, dataframe: pd.DataFrame = None, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize ML pipeline.

        Args:
            features_csv: Path to CSV file with extracted features
            dataframe: Optional pandas DataFrame to use directly (bypasses CSV load)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.features_csv = Path(features_csv) if features_csv else None
        self.dataframe = dataframe
        self.test_size = test_size
        self.random_state = random_state

        if self.features_csv:
            self.models_dir = self.features_csv.parent.parent / "models"
        else:
            self.models_dir = Path("models")

        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

        self.models = {}
        self.results = {}
        self.inference_times = {}
        self.feature_names = None
        self.selected_feature_names = None

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Load features from CSV or DataFrame and split into train/test sets.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.dataframe is not None:
            logger.info("Loading dataset from memory (DataFrame)...")
            df = self.dataframe.copy()
        elif self.features_csv and self.features_csv.exists():
            logger.info(f"Loading dataset from {self.features_csv}")
            df = pd.read_csv(self.features_csv)
        else:
            logger.error(f"Features file not found: {self.features_csv}")
            raise FileNotFoundError(f"Features CSV not found: {self.features_csv}")

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

        # Separate features and labels
        label_col = 'ground_truth_label'
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV")

        # USE SHARED FEATURE COLUMNS
        feature_cols = FEATURE_COLUMNS

        # Verify columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Columns missing from CSV: {missing_cols}")

        existing_cols = [col for col in feature_cols if col in df.columns]

        X = df[existing_cols]
        y = df[label_col]

        self.feature_names = existing_cols

        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Label distribution: {y.value_counts().to_dict()}")

        # Handle missing values
        X = X.fillna(X.mean())

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def perform_rfe(self, X_train: pd.DataFrame, y_train: pd.Series, pre_selected_features: List[str] = None) -> pd.Index:
        """
        Perform Recursive Feature Elimination to select top 20 features.
        Or use pre-selected features if provided.

        Args:
            X_train: Training features
            y_train: Training labels
            pre_selected_features: Optional list of feature names to use directly

        Returns:
            Index of selected feature names
        """
        logger.info("\n" + "="*60)
        logger.info("🧹 FEATURE SELECTION")
        logger.info("="*60)

        if pre_selected_features is not None:
            logger.info("Using pre-selected features (skipping RFE)...")
            selected_features = pd.Index(pre_selected_features)

            # Verify these features exist in X_train
            missing = [f for f in selected_features if f not in X_train.columns]
            if missing:
                logger.warning(f"⚠️ Some pre-selected features are missing from data: {missing}")
                selected_features = selected_features.intersection(X_train.columns)

            self.selected_feature_names = selected_features

            logger.info(f"Using {len(selected_features)} features:")
            for i, feat in enumerate(selected_features):
                logger.info(f"  {i+1}. {feat}")

            return selected_features

        logger.info("Running Recursive Feature Elimination (RFE)...")
        # Use Random Forest as estimator for RFE
        rf_estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )

        # Select top 20 features
        n_features_to_select = 20
        logger.info(f"Selecting top {n_features_to_select} features...")

        start_time = time.time()
        rfe = RFE(estimator=rf_estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X_train, y_train)
        rfe_time = time.time() - start_time

        # Get selected features
        selected_mask = rfe.support_
        selected_features = X_train.columns[selected_mask]
        self.selected_feature_names = selected_features

        logger.info(f"✅ RFE complete ({rfe_time:.2f}s)")
        logger.info(f"Selected Features ({len(selected_features)}):")
        for i, feat in enumerate(selected_features):
            logger.info(f"  {i+1}. {feat}")

        # Save selected features
        features_path = self.models_dir / "selected_features.pkl"
        joblib.dump(list(selected_features), features_path)
        logger.info(f"✅ Selected features saved to: {features_path}")

        return selected_features

    def train_svm_with_tuning(self, X_train: np.ndarray, X_test: np.ndarray) -> SVC:
        """
        Train SVM with hyperparameter tuning via RandomizedSearchCV.
        """
        logger.info("\n" + "="*60)
        logger.info("🤖 TRAINING: Support Vector Machine (SVM)")
        logger.info("="*60)

        param_dist_svm = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4],
        }

        svm_base = SVC(probability=True, random_state=self.random_state, class_weight='balanced')

        logger.info("Running RandomizedSearchCV for SVM (20 iterations)...")
        svm_search = RandomizedSearchCV(
            svm_base, param_dist_svm, n_iter=20, cv=3,
            scoring='f1_weighted', n_jobs=-1, random_state=self.random_state, verbose=1
        )

        start_time = time.time()
        svm_search.fit(X_train, self.y_train)
        train_time = time.time() - start_time

        logger.info(f"✅ SVM training complete ({train_time:.2f}s)")
        logger.info(f"   Best parameters: {svm_search.best_params_}")
        logger.info(f"   Best CV F1 score: {svm_search.best_score_:.4f}")

        svm_model = svm_search.best_estimator_
        self.models['SVM'] = svm_model

        return svm_model

    def train_rf_with_tuning(self, X_train: np.ndarray, X_test: np.ndarray) -> RandomForestClassifier:
        """
        Train Random Forest with hyperparameter tuning via RandomizedSearchCV.
        """
        logger.info("\n" + "="*60)
        logger.info("🤖 TRAINING: Random Forest (RF) - Tuning")
        logger.info("="*60)

        param_dist_rf = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
        }

        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, class_weight='balanced')

        logger.info("Running RandomizedSearchCV for RF (20 iterations)...")
        rf_search = RandomizedSearchCV(
            rf_base, param_dist_rf, n_iter=20, cv=3,
            scoring='f1_weighted', n_jobs=-1, random_state=self.random_state, verbose=1
        )

        start_time = time.time()
        rf_search.fit(X_train, self.y_train)
        train_time = time.time() - start_time

        logger.info(f"✅ RF training complete ({train_time:.2f}s)")
        logger.info(f"   Best parameters: {rf_search.best_params_}")
        logger.info(f"   Best CV F1 score: {rf_search.best_score_:.4f}")

        rf_model = rf_search.best_estimator_
        self.models['Random Forest'] = rf_model

        return rf_model

    def train_rf_fixed(self, X_train: np.ndarray, X_test: np.ndarray, params: Dict) -> RandomForestClassifier:
        """
        Train Random Forest with fixed hyperparameters (no tuning).
        """
        logger.info("\n" + "="*60)
        logger.info("🤖 TRAINING: Random Forest (RF) - Fixed Params")
        logger.info("="*60)
        logger.info(f"Params: {params}")

        rf_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced',
            **params
        )

        start_time = time.time()
        rf_model.fit(X_train, self.y_train)
        train_time = time.time() - start_time

        logger.info(f"✅ RF training complete ({train_time:.2f}s)")
        self.models['Random Forest'] = rf_model

        return rf_model

    def evaluate_model(self, model_name: str, model, X_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation with all metrics.
        """
        logger.info(f"\n📊 Evaluating {model_name}...")

        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        y_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr')

        fps = len(X_test) / inference_time if inference_time > 0 else float('inf')

        cm = confusion_matrix(self.y_test, y_pred)

        report = classification_report(self.y_test, y_pred, target_names=['Hover', 'Press', 'Hold', 'Release'])
        logger.info(f"\nClassification Report for {model_name}:\n{report}")

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fps': fps,
            'inference_time_ms': (inference_time / len(X_test)) * 1000,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }

        self.results[model_name] = metrics

        logger.info(f"✅ {model_name} Results:")
        logger.info(f"   Accuracy:    {accuracy:.4f}")
        logger.info(f"   Precision:   {precision:.4f}")
        logger.info(f"   Recall:      {recall:.4f}")
        logger.info(f"   F1-Score:    {f1:.4f}")
        logger.info(f"   ROC-AUC:     {roc_auc:.4f}")
        logger.info(f"   Inference:   {fps:.2f} FPS ({metrics['inference_time_ms']:.2f} ms/frame)")

        return metrics

    def compare_models(self) -> pd.DataFrame:
        """
        Create comprehensive comparison table of all models.
        """
        logger.info("\n" + "="*60)
        logger.info("📋 MODEL COMPARISON")
        logger.info("="*60)

        comparison_data = []

        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'FPS': f"{metrics['fps']:.2f}",
                'Inference (ms)': f"{metrics['inference_time_ms']:.2f}",
            })

        comparison_df = pd.DataFrame(comparison_data)

        logger.info("\n" + comparison_df.to_string(index=False))

        return comparison_df

    def visualize_results(self, output_dir: str = None) -> str:
        """
        Create visualization plots comparing models.
        """
        if output_dir is None:
            output_dir = self.features_csv.parent / "plots"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n📈 Generating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SVM vs Random Forest - Piano Motion Classification', fontsize=16, fontweight='bold')

        models_list = list(self.results.keys())
        class_names = ['Hover', 'Press', 'Hold', 'Release']

        for idx, model_name in enumerate(models_list[:2]):
            ax = axes[0, idx]
            cm = self.results[model_name]['confusion_matrix']

            im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
            ax.set_title(f'{model_name} - Confusion Matrix')

            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    text = ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')

            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')

        ax = axes[1, 0]

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_names))
        width = 0.35

        for idx, model_name in enumerate(models_list[:2]):
            metrics = self.results[model_name]
            values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
            ax.bar(x + idx*width, values, width, label=model_name, alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics Comparison')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        ax = axes[1, 1]

        fps_values = [self.results[name]['fps'] for name in models_list[:2]]
        colors = ['#FF6B6B', '#4ECDC4']

        bars = ax.bar(models_list[:2], fps_values, color=colors, alpha=0.8)
        ax.set_ylabel('Frames Per Second (FPS)')
        ax.set_title('Inference Speed Comparison')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        output_path = output_dir / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Comparison plot saved: {output_path}")

        plt.close('all')

        return str(output_dir)

    def run_pipeline(self, output_dir: str = None, selected_features: List[str] = None, fixed_rf_params: Dict = None, skip_svm: bool = False) -> str:
        """
        Execute complete ML pipeline.
        """
        if output_dir is None:
            if self.features_csv:
                output_dir = self.features_csv.parent.parent / "results"
            else:
                output_dir = Path("results")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING PIANOMOTION ML PIPELINE")
        logger.info("="*60)

        try:
            # Load and prepare data
            X_train_raw, X_test_raw, y_train, y_test = self.load_and_prepare_data()

            # Perform RFE (or use pre-selected)
            selected_features = self.perform_rfe(X_train_raw, y_train, pre_selected_features=selected_features)

            # Filter datasets to selected features
            X_train_selected = X_train_raw[selected_features]
            X_test_selected = X_test_raw[selected_features]

            # Scale features
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train_selected)
            X_test_scaled = self.scaler.transform(X_test_selected)

            # Save Scaler
            scaler_path = self.models_dir / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"✅ Scaler saved to: {scaler_path}")

            # Train models
            if not skip_svm:
                svm_model = self.train_svm_with_tuning(X_train_scaled, X_test_scaled)
                self.evaluate_model('SVM', svm_model, X_test_scaled)

            if fixed_rf_params:
                rf_model = self.train_rf_fixed(X_train_scaled, X_test_scaled, fixed_rf_params)
            else:
                rf_model = self.train_rf_with_tuning(X_train_scaled, X_test_scaled)

            # Save Random Forest Model (Preferred)
            rf_path = self.models_dir / "rf_model.pkl"
            joblib.dump(rf_model, rf_path)
            logger.info(f"✅ Random Forest model saved to: {rf_path}")

            # Evaluate models
            self.evaluate_model('Random Forest', rf_model, X_test_scaled)

            # Compare models
            comparison_df = self.compare_models()
            comparison_csv = output_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_csv, index=False)
            logger.info(f"✅ Comparison table saved: {comparison_csv}")

            # Visualize results
            self.visualize_results(str(output_dir))

            logger.info("\n" + "="*60)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info("="*60)
            logger.info(f"Results saved to: {output_dir}")

            return str(output_dir)

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Execute the complete ML pipeline."""

    # features.csv is now in data/
    features_csv = Path(__file__).parent.parent / "data" / "features.csv"
    output_dir = Path(__file__).parent.parent / "results"

    pipeline = PianoMotionMLPipeline(str(features_csv), test_size=0.2, random_state=42)

    try:
        results_dir = pipeline.run_pipeline(str(output_dir))
        print(f"\n🎉 All results saved to: {results_dir}")
    except FileNotFoundError:
        print("\n⚠️ Features file not found. Please run SyncPianoMotionDataset.py first.")

if __name__ == "__main__":
    main()
