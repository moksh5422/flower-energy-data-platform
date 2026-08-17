import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["asset_id", "timestamp"]).copy()

    # Calendar features
    x["hour"] = x["timestamp"].dt.hour
    x["dayofweek"] = x["timestamp"].dt.dayofweek

    # Historical lag features
    x["power_mw_lag_1"] = (
        x.groupby("asset_id")["power_mw"].shift(1)
    )

    x["power_mw_lag_24"] = (
        x.groupby("asset_id")["power_mw"].shift(24)
    )

    x["market_price_eur_mwh_lag_1"] = (
        x.groupby("asset_id")["market_price_eur_mwh"].shift(1)
    )

    # Historical rolling features.
    # shift(1) is critical: it prevents the current/future target
    # from leaking into the feature.
    x["power_roll_6"] = (
        x.groupby("asset_id")["power_mw"]
        .transform(lambda s: s.shift(1).rolling(6).mean())
    )

    x["price_roll_6"] = (
        x.groupby("asset_id")["market_price_eur_mwh"]
        .transform(lambda s: s.shift(1).rolling(6).mean())
    )

    return x.dropna().reset_index(drop=True)