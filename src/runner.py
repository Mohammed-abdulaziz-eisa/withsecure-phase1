"""
Main application script for running the binary classification ML model service.

This script initializes the ModelService, loads the ML model, makes
predictions based on predefined input parameters, and logs the output.
It demonstrates the typical workflow of using the ModelService in
a practical application context.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from config import settings
from loguru import logger
from model.model_service import ModelService
from model.pipeline.collection import load_test_data

# Add the src directory to Python path for proper imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@logger.catch
def main():
    """
    Run the application.

    Load the model, make predictions based on provided data,
    and log the prediction.
    """
    logger.info("Running the binary classification application...")

    # Initialize model service
    ml_svc = ModelService()
    ml_svc.load_model()

    # Load test data for batch prediction
    test_data = load_test_data()
    X_test = test_data.values.astype(np.float32)

    logger.info(f"Loaded test data: {X_test.shape}")

    # Make batch predictions
    predictions, probabilities = ml_svc.predict_batch(X_test)

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(settings.output_path, index=False, header=False)

    logger.info(f"Predictions saved to {settings.output_path}")
    logger.info("Prediction statistics:")
    logger.info(f"  Total predictions: {len(predictions)}")
    logger.info(
        f"  Class distribution: {dict(zip(*np.unique(predictions, return_counts=True), strict=False))}"
    )

    # Health check
    health_status = ml_svc.health_check()
    logger.info(f"Health Check: {health_status}")

    # Example single prediction
    sample_features = X_test[0].tolist()  # First sample
    single_prediction = ml_svc.predict(sample_features)
    logger.info(f"Sample prediction: {single_prediction}")

    logger.info("Application completed successfully!")


if __name__ == "__main__":
    main()
