import pandas as pd
import logging as log

class FetchData:
    """Fetches data from a path and returns a pandas dataframe."""

    def __init__(self, path: str):
        """Initializes the FetchData step.
        Args:
            path: Path to the data.
        """
        self.path = path
    
    def get_data(self):
        """Fetches data from a path and returns a pandas dataframe."""
        log.info(f'Fetching data from {self.path}')
        return pd.read_csv(self.path)