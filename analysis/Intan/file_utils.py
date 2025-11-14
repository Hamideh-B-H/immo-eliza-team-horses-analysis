import os
import pandas as pd

class FileManager():
    
    def __init__(self):
        pass
    
    def get_csv(self, file_path: str) -> pd.DataFrame:
        """
        Imports a CSV file into a pandas DataFrame.

        Automatically resolves the absolute path so it works
        regardless of folder depth.

        Parameters:
            file_path (str): Relative or absolute path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """

        # Resolve to an absolute, normalized path
        abs_path = os.path.abspath(os.path.expanduser(file_path))

        # Optional: check if file exists
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")

        df = pd.read_csv(abs_path)
        self.data = df
        
        return df
    
    def export_csv(self, df: pd.DataFrame, file_path: str, index: bool = False) -> None:
        """
        Exports a DataFrame to a CSV file.

        If df is None, exports the last loaded DataFrame (self.data).

        Parameters:
            df (pd.DataFrame, optional): DataFrame to export. Defaults to None (self.data).
            file_path (str): Path to save the CSV file. Defaults to 'output.csv'.
            index (bool): Whether to include row index in CSV. Defaults to False.
        """
        if df is None:
            if self.data is None:
                raise ValueError("No DataFrame provided and no DataFrame loaded in self.data.")
            df = self.data

        # Resolve absolute path
        abs_path = os.path.abspath(os.path.expanduser(file_path))

        # Export to CSV
        df.to_csv(abs_path, index=index)
        print(f"DataFrame exported to {abs_path}")

    
    