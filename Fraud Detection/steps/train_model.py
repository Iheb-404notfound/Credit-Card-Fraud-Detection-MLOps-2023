import tensorflow as tf
from typing import Tuple
import logging
from zenml import step
from src.model_training_strategy import TrainModel
import mlflow
from zenml.client import Client

experimentTracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experimentTracker.name)
def train_model(train_X: tf.Tensor, train_y: tf.Tensor, input_shape: Tuple[int] = (18,), num_classes: int = 1, to_load=True) -> tf.keras.Model:
    """Train the model on the given dataset.
    Args:
        train_X: Training features.
        train_y: Training labels.
        input_shape: Input shape of the model.
        num_classes: Number of output classes.
    Returns:
        model: Trained model.
    """
    try:
        mlflow.tensorflow.autolog() # automatically log metrics and parameters
        trainer = TrainModel(input_shape)
        mlflow.tensorflow.log_model(trainer.model, "model")
        return trainer.train(train_X, train_y, to_load=to_load)
    except Exception as e:
        logging.error(f'Training failed with error: {e}')
        return None