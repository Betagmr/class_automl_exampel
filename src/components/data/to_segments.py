def to_segments(df_data, column, size=24):
    start_idx = 24 - df_data.index.hour[0]
    df_data = df_data.iloc[start_idx:]
    val = df_data[[column]].to_numpy()

    return val[: size * (val.size // size)].reshape(-1, size)
