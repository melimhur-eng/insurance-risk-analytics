import pandas as pd


def load_data(path):
    """
    Load insurance dataset.
    """

    df = pd.read_csv(
        path,
        sep="|",
        low_memory=False
    )

    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(
            df["TransactionMonth"],
            errors="coerce"
        )

    return df