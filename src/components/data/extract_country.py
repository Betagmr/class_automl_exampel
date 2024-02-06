def extract_country(df_all, country_code, year_min=None, year_max=None):
    # List of columns to extract
    columns = [col for col in df_all.columns if col.startswith(country_code)]
    # Extract columns and remove country codes from column labels
    columns_map = {col: col[3:] for col in columns}
    df_out = df_all[columns].rename(columns=columns_map)

    # Exclude years outside of specified range, if any
    if year_min is not None:
        df_out = df_out[df_out.index.year >= year_min]
    if year_max is not None:
        df_out = df_out[df_out.index.year <= year_max]

    return df_out
