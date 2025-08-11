"""
Model service module for the binary classification pipeline.

This module provides a service layer for model management, prediction,
and artifact handling in production environments.
"""

import pickle as pk
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from config import settings
from loguru import logger

from .pipeline.model import BinaryClassificationPipeline, build_model


class ModelService:
    """
    A service class for managing the binary classification ML model.

    This class provides functionalities to load a ML model from
    a specified path, build it if it doesn't exist, and make
    predictions using the loaded model.

    Attributes:
        model: ML model managed by this service. Initially set to None.
        feature_processor: Feature preprocessing pipeline.
        pipeline: Complete ML pipeline with preprocessing and model.
    """

    def __init__(self) -> None:
        """Initialize the ModelService with no model loaded."""
        self.model = None
        self.feature_processor = None
        self.pipeline = None
        logger.info("ModelService initialized")

    def load_model(self, model_name: str | None = None) -> None:
        """
        Load the model from a specified path, or builds it if not exist.

        Args:
            model_name: Name of the model to load. If None, uses default from settings.
        """
        if model_name is None:
            model_name = settings.model_name

        logger.info(
            f"Checking the existence of model config file at "
            f"{settings.model_path}/{model_name}",
        )

        model_path = Path(f"{settings.model_path}/{model_name}")

        if not model_path.exists():
            logger.warning(
                f"Model at {model_path} was not found -> " f"building {model_name}",
            )
            build_model()

        logger.info(
            f"Model {model_name} exists! -> " "loading model configuration file",
        )

        try:
            # Try to load the latest pipeline artifact
            artifacts_dir = Path(settings.model_path)
            pipeline_files = list(artifacts_dir.glob("ml_pipeline_*.pkl"))

            if pipeline_files:
                # Load the most recent pipeline
                latest_pipeline = max(pipeline_files, key=lambda x: x.stat().st_mtime)
                artifacts = joblib.load(latest_pipeline)

                self.model = artifacts["model"]
                self.feature_processor = artifacts["feature_processor"]

                logger.info(f"Loaded pipeline from {latest_pipeline}")
            else:
                # Fallback to simple model loading
                with open(model_path, "rb") as model_file:
                    self.model = pk.load(model_file)

                logger.info(f"Loaded model from {model_path}")

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def predict(self, input_parameters: list[float]) -> list[int]:
        """
        Make a prediction using the loaded model.

        Args:
            input_parameters: List of input features for prediction.

        Returns:
            List containing the predicted class (-1 or 1).
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Making prediction!")

        try:
            # Convert to numpy array and reshape
            X = np.array(input_parameters).reshape(1, -1)

            # Apply preprocessing if available
            if self.feature_processor is not None:
                X = self.feature_processor.transform(X)

            # Make prediction
            prediction = self.model.predict(X)

            logger.debug(f"Single prediction made: {prediction[0]}")
            return prediction.tolist()

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def predict_batch(self, input_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Make batch predictions using the loaded model.

        Args:
            input_data: Numpy array of input features for batch prediction.

        Returns:
            Tuple of (predictions, probabilities) as numpy arrays.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        logger.info("Making batch predictions!")

        try:
            # Apply preprocessing if available
            if self.feature_processor is not None:
                input_data = self.feature_processor.transform(input_data)

            # Make predictions
            predictions = self.model.predict(input_data)
            probabilities = self.model.predict_proba(input_data)

            logger.info(f"Batch predictions completed for {len(predictions)} samples")
            return predictions, probabilities

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information.
        """
        if self.model is None:
            return {"status": "No model loaded"}

        try:
            model_info = {
                "model_type": type(self.model).__name__,
                "feature_processor_loaded": self.feature_processor is not None,
                "model_parameters": self.model.get_params()
                if hasattr(self.model, "get_params")
                else {},
            }

            # Add feature count if available
            if hasattr(self.model, "n_features_in_"):
                model_info["n_features"] = self.model.n_features_in_

            return model_info

        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"status": "Error getting model info", "error": str(e)}

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the model service.

        Returns:
            Dictionary containing health check results.
        """
        try:
            health_status = {
                "model_loaded": self.model is not None,
                "feature_processor_ready": self.feature_processor is not None,
                "model_info": self.get_model_info(),
                "status": "healthy",
            }

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "model_loaded": False,
                "feature_processor_ready": False,
                "model_info": {"status": "Error"},
                "status": "unhealthy",
                "error": str(e),
            }
