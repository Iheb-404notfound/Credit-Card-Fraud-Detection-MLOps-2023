import pandas as pd
import logging as log

class CleanData:
    """Clean data step class.
    Args:
        df: Pandas dataframe.
    Returns:
        Pandas dataframe.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean(self) -> pd.DataFrame:
        """Cleans data and returns a pandas dataframe.
        Returns:
            Pandas dataframe.
        """
        log.info('Cleaning data')
        self.df =  self.df.drop(columns=['id', 'V1', 'V3', 'V6', 'V7', 'V9', 'V10', 'V12', 'V14', 'V16', 'V17', 'V18']) \
                    .drop_duplicates()
        self.df /= self.df.max()
        return self.df