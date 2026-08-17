import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(
        ["asset_id", "timestamp"]
    ).copy()

    # Calendar features
    x["hour"] = x["timestamp"].dt.hour
    x["dayofweek"] = x["timestamp"].dt.dayofweek

    # Historical generation
    x["power_mw_lag_1"] = (
        x.groupby("asset_id")["power_mw"]
        .shift(1)
    )

    x["power_mw_lag_24"] = (
        x.groupby("asset_id")["power_mw"]
        .shift(24)
    )

    # Historical market price
    x["market_price_eur_mwh_lag_1"] = (
        x.groupby("asset_id")["market_price_eur_mwh"]
        .shift(1)
    )

    # Six-hour historical generation average
    x["power_roll_6"] = (
        x.groupby("asset_id")["power_mw"]
        .transform(
            lambda s: s.shift(1).rolling(6).mean()
        )
    )

    # Six-hour historical price average
    x["price_roll_6"] = (
        x.groupby("asset_id")["market_price_eur_mwh"]
        .transform(
            lambda s: s.shift(1).rolling(6).mean()
        )
    )

    # Remove rows where required model features don't exist.
    # state_of_charge is intentionally excluded because it is
    # not applicable to solar and wind assets.
    required_features = [
        "power_mw",
        "power_mw_lag_1",
        "power_mw_lag_24",
        "market_price_eur_mwh_lag_1",
        "power_roll_6",
        "price_roll_6",
    ]

    return x.dropna(
        subset=required_features
    ).reset_index(drop=True)