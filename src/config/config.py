"""
Configuration settings for the binary classification ML pipeline.

This module provides centralized configuration management using Pydantic settings,
supporting environment variable overrides for cloud deployment.
"""

import os

from loguru import logger
from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the ML pipeline."""

    model_config = SettingsConfigDict(
        env_file="src/config/.env", env_file_encoding="utf-8"
    )

    # Data paths
    train_data_path: str
    train_labels_path: str
    test_data_path: str
    output_path: str

    # Model settings
    model_path: str
    model_name: str

    # Logging
    log_level: str = "INFO"

    # Model hyperparameters
    random_state: int = 42
    n_cv_folds: int = 5
    lr_C: float = 0.1
    lr_max_iter: int = 1000

    # Artifacts and logs
    model_artifacts_dir: str = "./model_artifacts"
    logs_dir: str = "./logs"


# Initialize settings with robust fallback handling
def _initialize_settings() -> Settings:
    """Initialize settings with fallback values and proper path resolution."""
    try:
        # Try to load from environment first
        settings = Settings()
        logger.info("Settings loaded successfully from environment/config file")
        return settings
    except Exception as e:
        logger.warning(f"Failed to load settings from .env file: {e}")

        # Fallback to default values with proper path resolution
        from pathlib import Path

        # Determine data directory - check multiple possible locations to avoid errors
        data_dirs = ["./data", "../data", "data"]
        data_dir = None
        for dir_path in data_dirs:
            if Path(dir_path).exists():
                data_dir = dir_path
                break

        if data_dir is None:
            logger.warning("Data directory not found, using './data' as fallback")
            data_dir = "./data"

        # creating model directory
        model_dir = "./src/model/models"
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        settings = Settings(
            train_data_path=f"{data_dir}/train_data.csv",
            train_labels_path=f"{data_dir}/train_labels.csv",
            test_data_path=f"{data_dir}/test_data.csv",
            output_path=f"{data_dir}/test_labels.csv",
            model_path=model_dir,
            model_name="logistic_regression_v1",
            log_level="INFO",
            random_state=42,
            n_cv_folds=5,
            lr_C=0.1,
            lr_max_iter=1000,
        )

        logger.info(f"Fallback settings initialized with data_dir: {data_dir}")
        return settings


settings = _initialize_settings()

# Configure logging
logger.remove()
logger.add(
    f"{settings.logs_dir}/app.log",
    rotation="1 day",
    retention="2 days",
    compression="zip",
    level=settings.log_level,
)
