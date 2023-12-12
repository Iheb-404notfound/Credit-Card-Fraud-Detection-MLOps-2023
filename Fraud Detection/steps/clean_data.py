import logging as log
import pandas as pd
from src.data_cleaning_strategy import CleanData
from zenml import step



@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data step.
    Returns:
        Pandas dataframe.
    """
    try:
        return CleanData(df).clean()
    except Exception as e:
        log.error(f'Error cleaning data: {e}')
        raise e