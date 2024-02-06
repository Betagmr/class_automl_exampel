import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    df_data = pd.read_csv(
        "time_series_60min_singleindex.csv",
        index_col="utc_timestamp",
        parse_dates=True,
        low_memory=False,
    )

    return df_data
