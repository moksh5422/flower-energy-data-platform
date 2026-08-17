import numpy as np
import pandas as pd

from .config import RAW


def generate():
    rng = np.random.default_rng(42)

    ts = pd.date_range(
        "2026-01-01",
        periods=24 * 90,
        freq="h",
        tz="UTC",
    )

    assets = [
        "SOLAR_SE_01",
        "WIND_SE_01",
        "BATTERY_SE_01",
    ]

    rows = []

    for asset in assets:

        phase = rng.uniform(0, 2 * np.pi)

        # Asset-specific characteristics
        if asset == "SOLAR_SE_01":
            capacity_mw = 900
        elif asset == "WIND_SE_01":
            capacity_mw = 1100
        else:
            capacity_mw = 50

        for i, t in enumerate(ts):

            hour = t.hour
            day = t.dayofyear

            # ---------------------------------------------------------
            # Shared environmental conditions
            # ---------------------------------------------------------

            seasonal_temperature = (
                9
                + 11
                * np.sin(
                    2
                    * np.pi
                    * (day - 30)
                    / 365
                )
            )

            temperature = (
                seasonal_temperature
                + rng.normal(0, 2)
            )

            # Market conditions
            price = (
                72
                + 20
                * np.sin(
                    2
                    * np.pi
                    * (hour - 7)
                    / 24
                )
                + 12
                * np.sin(
                    2
                    * np.pi
                    * i
                    / 168
                )
                + rng.normal(0, 6)
            )

            grid = (
                0.4
                * np.sin(
                    2
                    * np.pi
                    * (hour - 4)
                    / 24
                )
                + rng.normal(0, 0.08)
            )

            # ---------------------------------------------------------
            # SOLAR
            # ---------------------------------------------------------

            if asset == "SOLAR_SE_01":

                daylight = max(
                    0,
                    np.sin(
                        np.pi
                        * (hour - 6)
                        / 12
                    ),
                )

                # Solar cloud pattern is independent from wind.
                cloud = np.clip(
                    0.35
                    + 0.30
                    * np.sin(
                        2
                        * np.pi
                        * i
                        / 72
                        + phase
                    )
                    + rng.normal(0, 0.12),
                    0,
                    1,
                )

                # Seasonal solar intensity
                seasonal_solar = (
                    0.85
                    + 0.15
                    * np.sin(
                        2
                        * np.pi
                        * (day - 80)
                        / 365
                    )
                )

                # Temperature efficiency
                temperature_factor = np.clip(
                    1
                    - 0.004
                    * (temperature - 25),
                    0.85,
                    1.10,
                )

                power = (
                    capacity_mw
                    * daylight
                    * seasonal_solar
                    * (1 - 0.85 * cloud)
                    * temperature_factor
                )

                power += rng.normal(0, 20)

                power = np.clip(
                    power,
                    0,
                    capacity_mw,
                )

                wind = max(
                    0,
                    6
                    + 2
                    * np.sin(
                        2
                        * np.pi
                        * i
                        / 168
                    )
                    + rng.normal(0, 1.5),
                )

                soc = np.nan

            # ---------------------------------------------------------
            # WIND
            # ---------------------------------------------------------

            elif asset == "WIND_SE_01":

                # Wind has its own weather process.
                wind = max(
                    0,
                    7.5
                    + 4
                    * np.sin(
                        2
                        * np.pi
                        * i
                        / 120
                        + phase
                    )
                    + 2
                    * np.sin(
                        2
                        * np.pi
                        * i
                        / 24
                    )
                    + rng.normal(0, 1.8),
                )

                # Cloud cover is deliberately less relevant to wind.
                cloud = np.clip(
                    0.45
                    + rng.normal(0, 0.20),
                    0,
                    1,
                )

                # Simplified turbine power curve
                cut_in = 3.0
                rated_speed = 12.0
                cut_out = 25.0

                if wind < cut_in:
                    wind_factor = 0

                elif wind < rated_speed:
                    wind_factor = (
                        (wind - cut_in)
                        / (rated_speed - cut_in)
                    ) ** 3

                elif wind <= cut_out:
                    wind_factor = 1

                else:
                    wind_factor = 0

                # Air-density correction
                air_density_factor = np.clip(
                    1
                    - 0.003
                    * (temperature - 15),
                    0.90,
                    1.10,
                )

                power = (
                    capacity_mw
                    * wind_factor
                    * air_density_factor
                )

                power += rng.normal(0, 25)

                power = np.clip(
                    power,
                    0,
                    capacity_mw,
                )

                soc = np.nan

            # ---------------------------------------------------------
            # BATTERY
            # ---------------------------------------------------------

            else:

                cloud = np.clip(
                    0.45
                    + rng.normal(0, 0.20),
                    0,
                    1,
                )

                wind = max(
                    0,
                    7
                    + rng.normal(0, 2),
                )

                soc = np.clip(
                    55
                    + 20
                    * np.sin(
                        2
                        * np.pi
                        * i
                        / 96
                    )
                    + rng.normal(0, 3),
                    5,
                    95,
                )

                # Battery dispatch responds to market/grid conditions.
                dispatch_signal = (
                    0.55 * (price - 72)
                    + 25 * grid
                )

                power = np.clip(
                    -dispatch_signal
                    + rng.normal(0, 4),
                    -capacity_mw,
                    capacity_mw,
                )

            rows.append(
                [
                    t,
                    asset,
                    temperature,
                    wind,
                    cloud,
                    price,
                    grid,
                    power,
                    soc,
                ]
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "asset_id",
            "temperature_c",
            "wind_speed_ms",
            "cloud_cover",
            "market_price_eur_mwh",
            "grid_imbalance",
            "power_mw",
            "state_of_charge",
        ],
    )

    df.to_csv(
        RAW / "energy_observations.csv",
        index=False,
    )

    return df


if __name__ == "__main__":
    generate()