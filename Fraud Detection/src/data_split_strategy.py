import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Tuple
from typing_extensions import Annotated
import logging as log

class DivideData:
    """Divide the data into training and evaluation sets."""
    def __init__(self, df:pd.DataFrame, test_size: float = 0.15):
        self.test_size = test_size
        self.df = df
    
    def divide(self) -> Tuple[
        Annotated[tf.Tensor, 'train_X'],
        Annotated[tf.Tensor, 'train_y'],
        Annotated[tf.Tensor, 'test_X'],
        Annotated[tf.Tensor, 'test_y']]:
        """Divide the data into training and evaluation sets.
            Returns:
                train_X, train_y, test_X, test_y
        """
        train_df, test_df = train_test_split(self.df, test_size=self.test_size)
        train_X, train_y = tf.constant(train_df.drop(columns=['Class']).to_numpy()), tf.constant(train_df['Class'])[..., tf.newaxis]
        test_X, test_y = tf.constant(test_df.drop(columns=['Class']).to_numpy()), tf.constant(test_df['Class'])[..., tf.newaxis]
        return train_X, train_y, test_X, test_y