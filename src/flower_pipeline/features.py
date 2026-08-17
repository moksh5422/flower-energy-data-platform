import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["asset_id","timestamp"]).copy()
    x["hour"] = x["timestamp"].dt.hour
    x["dayofweek"] = x["timestamp"].dt.dayofweek
    for col, lag in [("power_mw",1),("power_mw",24),("market_price_eur_mwh",1)]:
        name = f"{col.replace('_','')}_lag_{lag}"
        x[name] = x.groupby("asset_id")[col].shift(lag)
    x["power_roll_6"] = x.groupby("asset_id")["power_mw"].transform(
        lambda s: s.shift(1).rolling(6).mean()
    )
    x["price_roll_6"] = x.groupby("asset_id")["market_price_eur_mwh"].transform(
        lambda s: s.shift(1).rolling(6).mean()
    )
    return x.dropna().reset_index(drop=True)
