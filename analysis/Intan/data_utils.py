import pandas as pd
import re

class DataCleaner:

    @staticmethod
    def float_to_int(df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        Converts float columns to nullable integers (Int64).
        If `columns` is provided, only those columns are processed.
        """
        if columns is None:
            float_cols = df.select_dtypes(include=['float']).columns
        else:
            float_cols = [col for col in columns if col in df.columns]

        for col in float_cols:
            if pd.api.types.is_float_dtype(df[col]):
                # FIX: Use round without multiplying and safely convert to Int64
                df[col] = df[col].round(0)  # just round to nearest whole number
                df[col] = df[col].astype("Int64")  # convert to nullable integer
        return df

    @staticmethod
    def extract_int_from_string(df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        Converts string columns containing numeric values into integers.
        Pure text or URL columns are left untouched.
        If `columns` is provided, only those columns are processed.
        """
        if columns is None:
            str_cols = df.select_dtypes(include=['object']).columns
        else:
            str_cols = [col for col in columns if col in df.columns]

        for col in str_cols:
            # Only process columns that contain at least one numeric string
            if df[col].dropna().astype(str).str.contains(r"\d").any():
                df[col] = df[col].apply(
                    lambda x: int(re.sub(r"[^\d]", "", str(x)))
                    if pd.notnull(x) and re.search(r"\d", str(x))
                    else pd.NA
                )
        return df

    @staticmethod
    def auto_clean(df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        Cleans selected columns of the DataFrame:
        - Converts floats to integers
        - Extracts integers from numeric strings
        Columns not in `columns` remain untouched.
        """
        df = DataCleaner.float_to_int(df, columns=columns)
        df = DataCleaner.extract_int_from_string(df, columns=columns)
        return df
