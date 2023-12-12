import tensorflow as tf
from typing import Tuple, Dict


class EvaluateModel:
    """Evaluate the model."""

    def __init__(self, model: tf.keras.Model):
        self.model = model
    
    def evaluate(self, test_X: tf.Tensor, test_y: tf.Tensor, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate the model on the given dataset.
        Args:
            test_X: Test features.
            test_y: Test labels.
            batch_size: Batch size for evaluation.
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        loss, accuracy = self.model.evaluate(dataset)
        return {'loss': loss, 'accuracy': accuracy}
