from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated
import tensorflow as tf
from zenml import step
from src.data_split_strategy import DivideData


@step
def divide_data(df: pd.DataFrame, test_size: float = 0.15) -> Tuple[
    Annotated[tf.Tensor, 'train_X'],
    Annotated[tf.Tensor, 'train_y'],
    Annotated[tf.Tensor, 'test_X'],
    Annotated[tf.Tensor, 'test_y']]:
    """Divide the data into training and evaluation sets.
        Args:
            df: Input dataframe.
            test_size: Size of the test set.
        Returns:
            train_X, train_y, test_X, test_y
    """
    train_X, train_y, test_X, test_y = DivideData(df, test_size).divide()
    return train_X, train_y, test_X, test_y