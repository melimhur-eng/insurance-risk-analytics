import pandas as pd


def clean_data(df):
    """
    Basic data cleaning.
    """

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing categorical values
    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Fill numeric missing values
    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(
            df[col].median()
        )

    # Create Loss Ratio
    df["LossRatio"] = (
        df["TotalClaims"] /
        (df["TotalPremium"] + 1e-6)
    )

    # Create Margin
    df["Margin"] = (
        df["TotalPremium"] -
        df["TotalClaims"]
    )

    return df