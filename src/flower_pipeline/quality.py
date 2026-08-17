import pandas as pd

REQUIRED = {
    "timestamp","asset_id","temperature_c","wind_speed_ms","cloud_cover",
    "market_price_eur_mwh","grid_imbalance","power_mw","state_of_charge"
}

def validate(df: pd.DataFrame) -> dict:
    missing = REQUIRED - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df["timestamp"].notna().all()
    assert df["asset_id"].notna().all()
    assert df["cloud_cover"].between(0,1).all()
    assert df["market_price_eur_mwh"].notna().all()
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated(["timestamp","asset_id"]).sum()),
        "null_cells": int(df.isna().sum().sum())
    }
