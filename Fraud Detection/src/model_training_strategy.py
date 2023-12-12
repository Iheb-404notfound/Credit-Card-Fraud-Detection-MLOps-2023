import tensorflow as tf
from typing import Tuple
import logging as log


class TrainModel:
    """Train The Model."""

    def __init__(self, input_shape: Tuple[int, ...]=(18,), num_classes: int=1):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='sigmoid')
        ])
    
    def train(self, train_X: tf.Tensor=None, train_y: tf.Tensor=None, batch_size: int = 32, epochs: int = 10, to_load=False) -> tf.keras.Model:
        """Train the model on the given dataset.
        Args:
            train_X: Training features.
            train_y: Training labels.
            batch_size: Batch size for training.
            epochs: Number of epochs to train for.
        Returns:
            model: Trained model.
        """
        if to_load:
            try:
                self.model.load_weights('./saved_model/weights.keras')
                log.info('Loaded weights from previous training.')
            except Exception as e:
                log.error(f'Loading weights failed with error: {e}')
            return self.model

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self.model.fit(dataset, epochs=epochs, verbose=0, callbacks=[tf.keras.callbacks.ModelCheckpoint('./saved_model/weights.keras', save_best_only=True, save_weights_only=True, monitor='accuracy')])
        return self.model