"""
Data collection module for the binary classification pipeline.

This module handles loading and validation of training and test datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from config import settings
from loguru import logger


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        path: Path to the CSV file. If None, uses default from settings.

    Returns:
        DataFrame containing the loaded data.
    """
    if path is None:
        path = settings.train_data_path

    logger.info(f"Loading data from {path}")

    try:
        data = pd.read_csv(path, header=None)
        logger.info(f"Data loaded successfully: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {str(e)}")
        raise


def load_labels(path: str = None) -> np.ndarray:
    """
    Load labels from CSV file.

    Args:
        path: Path to the labels CSV file. If None, uses default from settings.

    Returns:
        Numpy array containing the labels.
    """
    if path is None:
        path = settings.train_labels_path

    logger.info(f"Loading labels from {path}")

    try:
        labels = pd.read_csv(path, header=None).values.ravel()
        logger.info(f"Labels loaded successfully: {labels.shape}")
        return labels
    except Exception as e:
        logger.error(f"Failed to load labels from {path}: {str(e)}")
        raise


def load_test_data(path: str = None) -> pd.DataFrame:
    """
    Load test data from CSV file.

    Args:
        path: Path to the test data CSV file. If None, uses default from settings.

    Returns:
        DataFrame containing the test data.
    """
    if path is None:
        path = settings.test_data_path

    logger.info(f"Loading test data from {path}")

    try:
        data = pd.read_csv(path, header=None)
        logger.info(f"Test data loaded successfully: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load test data from {path}: {str(e)}")
        raise


def validate_data_files() -> bool:
    """
    Validate that all required data files exist.

    Returns:
        True if all files exist, False otherwise.
    """
    required_files = [
        settings.train_data_path,
        settings.train_labels_path,
        settings.test_data_path,
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            return False

    logger.info("All required data files found")
    return True
