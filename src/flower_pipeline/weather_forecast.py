import numpy as np
import pandas as pd

from .config import RAW


def generate_forecast():
    rng = np.random.default_rng(123)

    df = pd.read_csv(
        RAW / "energy_observations.csv",
        parse_dates=["timestamp"],
    )

    # Forecast weather is created from the actual weather
    # plus realistic forecast uncertainty.
    forecast = df[
        [
            "timestamp",
            "asset_id",
            "temperature_c",
            "wind_speed_ms",
            "cloud_cover",
        ]
    ].copy()

    forecast["temperature_forecast_c"] = (
        forecast["temperature_c"]
        + rng.normal(0, 1.0, len(forecast))
    )

    forecast["wind_speed_forecast_ms"] = np.clip(
        forecast["wind_speed_ms"]
        + rng.normal(0, 1.2, len(forecast)),
        0,
        None,
    )

    forecast["cloud_cover_forecast"] = np.clip(
        forecast["cloud_cover"]
        + rng.normal(0, 0.12, len(forecast)),
        0,
        1,
    )

    # Remove actual weather fields.
    forecast = forecast[
        [
            "timestamp",
            "asset_id",
            "temperature_forecast_c",
            "wind_speed_forecast_ms",
            "cloud_cover_forecast",
        ]
    ]

    forecast.to_csv(
        RAW / "weather_forecast.csv",
        index=False,
    )

    print(
        f"Weather forecast generated: {len(forecast)} rows"
    )

    return forecast


if __name__ == "__main__":
    generate_forecast()