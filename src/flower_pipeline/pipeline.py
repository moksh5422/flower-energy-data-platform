import pandas as pd
import duckdb
from .config import RAW, BRONZE, SILVER, GOLD, DATA
from .quality import validate

def run():
    src = pd.read_csv(RAW/"energy_observations.csv", parse_dates=["timestamp"])
    metrics = validate(src)

    bronze = src.copy()
    bronze["ingested_at"] = pd.Timestamp.utcnow()
    bronze.to_parquet(BRONZE/"energy_observations.parquet", index=False)

    silver = bronze.sort_values(["asset_id","timestamp"]).drop_duplicates(
        ["asset_id","timestamp"], keep="last"
    )
    silver["asset_type"] = silver["asset_id"].str.split("_").str[0]
    silver.to_parquet(SILVER/"energy_observations.parquet", index=False)

    gold = silver.groupby(["timestamp","asset_type"], as_index=False).agg(
        total_power_mw=("power_mw","sum"),
        avg_market_price=("market_price_eur_mwh","mean"),
        avg_temperature_c=("temperature_c","mean"),
        avg_wind_speed_ms=("wind_speed_ms","mean"),
        avg_cloud_cover=("cloud_cover","mean"),
        avg_grid_imbalance=("grid_imbalance","mean")
    )
    gold.to_parquet(GOLD/"asset_hourly.parquet", index=False)

    con = duckdb.connect(str(DATA/"flower.duckdb"))
    con.execute("CREATE OR REPLACE TABLE silver_energy AS SELECT * FROM read_parquet(?)",
                [str(SILVER/"energy_observations.parquet")])
    con.execute("CREATE OR REPLACE TABLE gold_asset_hourly AS SELECT * FROM read_parquet(?)",
                [str(GOLD/"asset_hourly.parquet")])
    con.close()
    print("Pipeline complete:", metrics)

if __name__ == "__main__":
    run()
