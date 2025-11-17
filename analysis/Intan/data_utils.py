import pandas as pd
import re

class DataCleaner:

    @staticmethod
    def float_to_int(df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        Converts float columns to nullable integers (Int64).
        If `columns` is provided, only those columns are processed.
        """

        # 🔥 FIX #1: include float64 dtype correctly
        if columns is None:
            float_cols = df.select_dtypes(include=['float', 'float64']).columns
        else:
            float_cols = [col for col in columns if col in df.columns and pd.api.types.is_float_dtype(df[col])]

        for col in float_cols:
            df[col] = df[col].round(0)
            df[col] = df[col].astype("Int64")
        return df


    @staticmethod
    def extract_int_from_string(df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        Converts string columns containing numeric values into integers.
        Pure text or URL columns are left untouched.
        Only extracts a single numeric value from the string:
        - Price on request -> NA
        - Make offer from 290 000 € -> 290000
        - 149 000 € -> 149000
        """

        if columns is None:
            str_cols = df.select_dtypes(include=['object']).columns
        else:
            str_cols = [col for col in columns if col in df.columns]

        for col in str_cols:
            if df[col].dropna().astype(str).str.contains(r"\d").any():

                def extract_single_number(x):
                    if pd.isna(x):
                        return pd.NA

                    s = str(x)

                    # match first standalone number (with spaces or special spaces)
                    match = re.search(r"\d[\d \u202f\u00a0]*", s)
                    if not match:
                        return pd.NA

                    num = match.group(0)

                    # remove all spaces / NBSPs
                    num = re.sub(r"[\s\u202f\u00a0]+", "", num)

                    return int(num) if num.isdigit() else pd.NA

                df[col] = df[col].apply(extract_single_number)

        return df



    @staticmethod
    def auto_clean(df: pd.DataFrame, columns=None) -> pd.DataFrame:
        df = DataCleaner.float_to_int(df, columns=columns)
        df = DataCleaner.extract_int_from_string(df, columns=columns)
        return df
