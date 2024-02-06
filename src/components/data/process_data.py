import numpy as np
from stac.data_processing.add_time_features import add_time_features

from src.components.data.extract_country import extract_country
from src.components.data.to_segments import to_segments
from src.components.data.transform_dataframe import transform_dataframe


def process_data(df_data):
    df_hourly = extract_country(
        df_data,
        country_code="BE",
        year_min=2015,
        year_max=2019,
    )

    cols_map = {
        "load_actual_entsoe_transparency": "Consumption",
        "wind_generation_actual": "Wind",
        "solar_generation_actual": "Solar",
    }

    df_be = transform_dataframe(df_hourly, cols_map).dropna()
    df_be["Wind+Solar"] = df_be[["Wind", "Solar"]].sum(axis=1, skipna=False)
    df_be = add_time_features(df_be, include=["season"])

    spring = to_segments(df_be.query("season == 0"), "Consumption", size=24)
    summer = to_segments(df_be.query("season == 1"), "Consumption", size=24)
    autumn = to_segments(df_be.query("season == 2"), "Consumption", size=24)
    winter = to_segments(df_be.query("season == 3"), "Consumption", size=24)

    x_data = np.concatenate((winter, summer, autumn, spring))
    y_data = np.concatenate(
        (
            0 * np.ones(winter.shape[0]),
            1 * np.ones(summer.shape[0]),
            2 * np.ones(autumn.shape[0]),
            3 * np.ones(spring.shape[0]),
        )
    )

    return x_data, y_data
