import pandas as pd
import numpy as np

# -----------------------------
# Data Cleaning Script for Immo Eliza
# -----------------------------

def clean_immo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full data cleaning on the Immo Eliza dataset.
    Steps:
        - Strip whitespace
        - Remove duplicates
        - Fix data types
        - Clean numeric columns
        - Remove rows without province
        - Compute price_per_m2
        - Remove outliers for price and living_area
    """

    df = df.copy()

    # --------------------------------------
    # 1. Strip whitespace from string columns
    # --------------------------------------
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan})

    # ----------------------
    # 2. Remove duplicates
    # ----------------------
    df.drop_duplicates(subset=["property_id"], inplace=True)

    # ----------------------
    # 3. Fix data types
    # ----------------------
    numeric_columns = ["price", "living_area", "number_rooms", "facades"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Postal code should be string
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype("Int64").astype(str)

    # ----------------------
    # 4. Remove rows without province
    # ----------------------
    df["province"] = df["province"].replace({"": np.nan})
    df = df[df["province"].notna()].copy()

    # ----------------------
    # 5. Compute price_per_m2
    # ----------------------
    df["price_per_m2"] = df["price"] / df["living_area"]

    # ----------------------
    # 6. Remove outliers using IQR
    # ----------------------
    def remove_outliers(series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return series.where(series.between(lower, upper))

    df["price"] = remove_outliers(df["price"])
    df["living_area"] = remove_outliers(df["living_area"])

    # Remove rows where these became NaN after outlier filtering
    df.dropna(subset=["price", "living_area"], inplace=True)

    return df


# Example usage:
df = pd.read_csv("Hamideh_final_data_cleaned.csv")
df_clean = clean_immo_data(df)
df_clean.to_csv("cleaned_immo_eliza.csv", index=False)
