import pandas as pd
import logging as log
from zenml import step
from src.data_loading_strategy import FetchData


@step
def fetch_data(path: str) -> pd.DataFrame:
    """Fetches data from a path and returns a pandas dataframe.
    Args:
        path: Path to the data.
    Returns:
        Pandas dataframe.
    """
    try:
        return FetchData(path).get_data()
    except Exception as e:
        log.error(f'Unable to fetch data from {path}')
        raise e