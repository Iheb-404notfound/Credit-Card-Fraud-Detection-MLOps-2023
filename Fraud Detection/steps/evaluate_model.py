from zenml import step
import numpy as np
from typing import Dict
import tensorflow as tf
import logging
from src.model_evaluation_strategy import EvaluateModel
import mlflow
from zenml.client import Client

experiement_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiement_tracker.name)
def evaluate_model(model: tf.keras.Model, test_X: tf.Tensor, test_y: tf.Tensor) -> Dict[str, float]:
    """Evaluate the model on the given dataset.
    Args:
        model: Trained model.
        test_X: Test features.
        test_y: Test labels.
    Returns:
        Dictionary of evaluation metrics.
    """
    try:
        evaluator = EvaluateModel(model)
        results = evaluator.evaluate(test_X, test_y)
        mlflow.log_metrics(results)
        # logging.info(f'Evaluation results: {results}')
        return results
    except Exception as e:
        logging.error(f'Evaluation failed with error: {e}')
        return {'loss': np.nan, 'accuracy': np.nan}