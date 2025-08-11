"""
Data preparation module for the binary classification pipeline.

This module handles data preprocessing including SMOTE resampling
and robust scaling for consistent preprocessing in production.
"""

import numpy as np
from config import settings
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.preprocessing import RobustScaler


class FeatureProcessor:
    """
    Production feature preprocessing pipeline.

    Handles SMOTE resampling for imbalance correction and robust scaling.
    """

    def __init__(self, random_state: int | None = None):
        """
        Initialize the feature processor.

        Args:
            random_state: Random seed for reproducibility.
        """
        if random_state is None:
            random_state = settings.random_state

        self.scaler = RobustScaler()
        self.smote = SMOTE(random_state=random_state, k_neighbors=5)
        self._is_fitted = False
        self._train_distribution = None

        logger.info("FeatureProcessor initialized")

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessing pipeline with scaling and custom SMOTE.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Tuple of (X_resampled, y_resampled) - the processed and resampled data
        """
        try:
            logger.info("Fitting feature preprocessing pipeline...")

            # Scale the data first
            X_scaled = self.scaler.fit_transform(X)

            # Apply SMOTE for class imbalance
            X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)

            self._is_fitted = True
            self._train_distribution = X_scaled.mean(axis=0)  # Store mean as baseline

            # Log preprocessing statistics
            n_original = X.shape[0]
            n_resampled = X_resampled.shape[0]
            logger.info("Feature preprocessing completed:")
            logger.info(f"  Original samples: {n_original}")
            logger.info(f"  Resampled samples: {n_resampled}")
            logger.info(
                f"  Original class distribution: {dict(zip(*np.unique(y, return_counts=True), strict=False))}"
            )
            logger.info(
                f"  Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True), strict=False))}"
            )

            return X_resampled, y_resampled
        except Exception as e:
            logger.error(f"FeatureProcessor fit failed: {str(e)}")
            raise

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted preprocessing pipeline to test data (scaling only, no SMOTE).

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if not self._is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")

        try:
            return self.scaler.transform(X)
        except Exception as e:
            logger.error(f"FeatureProcessor transform failed: {str(e)}")
            raise


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for training and testing.

    Returns:
        Tuple of (X_train, y_train, X_test) as numpy arrays
    """
    logger.info("Starting data preparation...")

    try:
        # Load data
        from .collection import load_data, load_labels, load_test_data

        train_data = load_data()
        train_labels = load_labels()
        test_data = load_test_data()

        # Convert to numpy arrays
        X_train = train_data.values.astype(np.float32)
        y_train = train_labels.astype(np.int32)
        X_test = test_data.values.astype(np.float32)

        # Validate data and get potentially converted labels
        y_train = _validate_data(X_train, y_train, X_test)

        logger.info("Data preparation completed:")
        logger.info(
            f"  Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}"
        )
        logger.info(f"  Test samples: {X_test.shape[0]}")
        logger.info(
            f"  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True), strict=False))}"
        )

        return X_train, y_train, X_test

    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise


def _validate_data(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """
    Validate prepared data for consistency and quality.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features

    Returns:
        Validated and possibly converted y_train labels
    """
    try:
        # Check for missing values
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("Data contains missing values")

        # Check feature consistency
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Training and test data have different number of features")

        # Check and convert label format
        unique_labels = np.unique(y_train)
        expected_labels = [-1, 1]
        if not np.array_equal(np.sort(unique_labels), expected_labels):
            logger.info(f"Converting label format from {unique_labels} to [-1, 1]")
            # Convert 0/1 labels to -1/1 format if needed
            if np.array_equal(np.sort(unique_labels), [0, 1]):
                y_train = np.where(y_train == 0, -1, 1)
                logger.info("Successfully converted [0, 1] labels to [-1, 1] format")
            else:
                logger.warning(f"Unexpected label format: {unique_labels}")

        # Check sample consistency
        if len(y_train) != X_train.shape[0]:
            raise ValueError(
                "Number of labels doesn't match number of training samples"
            )

        logger.info("Data validation passed")
        return y_train

    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise
