import numpy as np
import pandas as pd
from .config import RAW

def generate():
    rng = np.random.default_rng(42)
    ts = pd.date_range("2026-01-01", periods=24*90, freq="h", tz="UTC")
    assets = ["SOLAR_SE_01", "WIND_SE_01", "BATTERY_SE_01"]
    rows = []

    for asset in assets:
        phase = rng.uniform(0, 2*np.pi)
        for i, t in enumerate(ts):
            hour = t.hour
            day = t.dayofyear
            temperature = 7 + 10*np.sin(2*np.pi*(day-30)/365) + rng.normal(0,2)
            wind = max(0, 7 + 3*np.sin(2*np.pi*i/168 + phase) + rng.normal(0,1.5))
            cloud = np.clip(0.45 + 0.35*np.sin(2*np.pi*i/48 + phase) + rng.normal(0,.12),0,1)
            solar = max(0,np.sin(np.pi*(hour-6)/12)) if 6 <= hour <= 18 else 0
            price = 72 + 20*np.sin(2*np.pi*(hour-7)/24) + 12*np.sin(2*np.pi*i/168) + rng.normal(0,7)
            grid = .4*np.sin(2*np.pi*(hour-4)/24) + rng.normal(0,.08)

            if asset.startswith("SOLAR"):
                power = max(0, 900*solar*(1-cloud) + rng.normal(0,35))
                soc = np.nan
            elif asset.startswith("WIND"):
                power = max(0, 650*(wind/12)**3 + rng.normal(0,25))
                soc = np.nan
            else:
                power = rng.normal(0,10)
                soc = np.clip(55 + 20*np.sin(2*np.pi*i/96) + rng.normal(0,2),5,95)

            rows.append([t,asset,temperature,wind,cloud,price,grid,power,soc])

    df = pd.DataFrame(rows, columns=[
        "timestamp","asset_id","temperature_c","wind_speed_ms","cloud_cover",
        "market_price_eur_mwh","grid_imbalance","power_mw","state_of_charge"
    ])
    df.to_csv(RAW/"energy_observations.csv", index=False)
    return df

if __name__ == "__main__":
    generate()
