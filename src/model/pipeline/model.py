"""
Model training module for the binary classification pipeline.

This module handles model training, validation, and artifact management
for the Logistic Regression binary classifier.
"""

import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psutil
from config import settings
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .preparation import FeatureProcessor, prepare_data


@dataclass
class ModelMetrics:
    """Data class for storing model performance metrics."""

    model_name: str
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    training_time: float
    n_features_used: int
    n_training_samples: int
    timestamp: str | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BinaryClassificationPipeline:
    """
     Production ML pipeline for binary classification using Logistic Regression.

    The following steps are implemented in this pipeline:
     - Model versioning and serialization
     - Comprehensive logging and monitoring
     - Performance validation
     - Error handling and recovery
     - Artifact management
     - Memory efficiency and health monitoring
    """

    def __init__(self):
        """Initialize the ML pipeline."""
        self.feature_processor = None
        self.model = None
        self.metrics = {}
        self.is_trained = False
        self._last_training_time = None
        self._previous_model = None

        # Set random seeds for reproducibility
        np.random.seed(settings.random_state)

        # Create directories
        self._create_directories()

        logger.info("BinaryClassificationPipeline initialized")

    def _create_directories(self) -> None:
        """Create necessary directories for artifacts."""
        directories = [settings.model_path, settings.logs_dir]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

    def _log_memory_usage(self) -> None:
        """Log current memory usage for monitoring."""
        memory = psutil.virtual_memory()
        logger.info(f"Memory usage: {memory.percent}%")

    def train_and_validate(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train Logistic Regression model with cross-validation and performance monitoring.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("Starting model training and validation...")
        self._log_memory_usage()

        try:
            if X_train.size == 0 or y_train.size == 0:
                raise ValueError("Empty training data or labels")

            # Initialize feature processor
            self.feature_processor = FeatureProcessor()

            # Fit preprocessing pipeline (includes SMOTE resampling)
            X_train_processed, y_train_processed = self.feature_processor.fit(
                X_train, y_train
            )

            # Initialize model with hyperparameters
            self.model = LogisticRegression(
                C=settings.lr_C,
                max_iter=settings.lr_max_iter,
                random_state=settings.random_state,
                class_weight="balanced",
            )

            # Train with cross-validation
            start_time = time.time()

            cv = StratifiedKFold(
                n_splits=settings.n_cv_folds,
                shuffle=True,
                random_state=settings.random_state,
            )
            cv_scores = cross_val_score(
                self.model,
                X_train_processed,
                y_train_processed,
                cv=cv,
                scoring="roc_auc",
            )

            # Train final model on full processed dataset
            self.model.fit(X_train_processed, y_train_processed)

            training_time = time.time() - start_time
            self._last_training_time = training_time

            # Store metrics
            self.metrics = ModelMetrics(
                model_name=settings.model_name,
                cv_roc_auc_mean=cv_scores.mean(),
                cv_roc_auc_std=cv_scores.std(),
                training_time=training_time,
                n_features_used=X_train.shape[1],
                n_training_samples=X_train.shape[0],
            )

            self.is_trained = True

            logger.info("Logistic Regression trained successfully:")
            logger.info(
                f"  ROC-AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
            )
            logger.info(f"  Training time: {training_time:.2f}s")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model.

        Args:
            X_test: Test features

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Apply preprocessing
            X_test_processed = self.feature_processor.transform(X_test)

            # Make predictions
            predictions = self.model.predict(X_test_processed)
            probabilities = self.model.predict_proba(X_test_processed)

            # Validate predictions
            self._validate_predictions(predictions, probabilities)

            # Log prediction statistics
            self._log_prediction_stats(predictions, probabilities)

            return predictions, probabilities

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _validate_predictions(
        self, predictions: np.ndarray, probabilities: np.ndarray
    ) -> None:
        """
        Validate prediction outputs.

        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
        """
        try:
            # Check prediction shape
            if predictions.size == 0:
                raise ValueError("Empty predictions")

            # Check probability shape
            if probabilities.size == 0:
                raise ValueError("Empty probabilities")

            # Check for valid class labels
            unique_predictions = np.unique(predictions)
            valid_label_sets = [[-1, 1], [0, 1]]
            is_valid = any(
                all(pred in valid_set for pred in unique_predictions)
                for valid_set in valid_label_sets
            )
            if not is_valid:
                logger.warning(f"Unexpected prediction labels: {unique_predictions}")
            else:
                logger.debug(f"Prediction labels validated: {unique_predictions}")

            # Check probability sums
            prob_sums = np.sum(probabilities, axis=1)
            if not np.allclose(prob_sums, 1.0, atol=1e-6):
                raise ValueError("Probability sums must equal 1.0")

            logger.info("Prediction validation passed")

        except Exception as e:
            logger.error(f"Prediction validation failed: {str(e)}")
            raise

    def _log_prediction_stats(
        self, predictions: np.ndarray, probabilities: np.ndarray
    ) -> None:
        """
        Log prediction statistics.

        Args:
            predictions: Model predictions
            probabilities: Prediction probabilities
        """
        try:
            n_predictions = len(predictions)
            class_counts = dict(
                zip(*np.unique(predictions, return_counts=True), strict=False)
            )
            avg_prob = np.mean(probabilities, axis=0)

            logger.info("Prediction statistics:")
            logger.info(f"  Total predictions: {n_predictions}")
            logger.info(f"  Class distribution: {class_counts}")
            logger.info(f"  Average probabilities: {avg_prob}")

        except Exception as e:
            logger.error(f"Failed to log prediction stats: {str(e)}")

    def save_artifacts(self) -> None:
        """Save model artifacts with timestamp."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving artifacts")

        logger.info("Saving model artifacts...")
        self._log_memory_usage()

        try:
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Prepare artifacts
            artifacts = {
                "model": self.model,
                "feature_processor": self.feature_processor,
                "metrics": asdict(self.metrics),
                "config": {
                    "model_name": settings.model_name,
                    "random_state": settings.random_state,
                    "lr_C": settings.lr_C,
                    "lr_max_iter": settings.lr_max_iter,
                    "n_cv_folds": settings.n_cv_folds,
                },
                "timestamp": timestamp,
            }

            # Save to file
            artifact_path = Path(settings.model_path) / f"ml_pipeline_{timestamp}.pkl"
            joblib.dump(artifacts, artifact_path)

            logger.info(f"Artifacts saved to {artifact_path}")
            logger.info("Model metrics:")
            logger.info(
                f"  ROC-AUC: {self.metrics.cv_roc_auc_mean:.4f} (+/- {self.metrics.cv_roc_auc_std:.4f})"
            )
            logger.info(f"  Training time: {self.metrics.training_time:.2f}s")
            logger.info(f"  Features used: {self.metrics.n_features_used}")
            logger.info(f"  Training samples: {self.metrics.n_training_samples}")

        except Exception as e:
            logger.error(f"Failed to save artifacts: {str(e)}")
            raise

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on the pipeline.

        Returns:
            Dictionary containing health status
        """
        try:
            return {
                "is_trained": self.is_trained,
                "model_loaded": self.model is not None,
                "feature_processor_ready": self.feature_processor is not None,
                "last_training_time": self._last_training_time,
                "metrics_available": bool(self.metrics),
                "status": "healthy",
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}


def build_model() -> None:
    """Build and train the ML model."""
    logger.info("Starting model building process...")

    try:
        # Prepare data
        X_train, y_train, X_test = prepare_data()

        # Initialize and train pipeline
        pipeline = BinaryClassificationPipeline()
        pipeline.train_and_validate(X_train, y_train)

        # Save artifacts
        pipeline.save_artifacts()

        logger.info("Model building completed successfully")

    except Exception as e:
        logger.error(f"Model building failed: {str(e)}")
        raise
