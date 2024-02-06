def transform_dataframe(df_data, cols_map):
    # Rename columns for convenience
    df_data = df_data[list(cols_map.keys())].rename(columns=cols_map)
    df_data = df_data / 1000  # Convert from MW to GW
    df_data = df_data.rename_axis("Date")
    return df_data
