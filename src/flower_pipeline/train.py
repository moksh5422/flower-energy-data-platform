import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from .config import SILVER, GOLD, MODELS
from .features import make_features

FEATURES = [
    "temperature_c","wind_speed_ms","cloud_cover","market_price_eur_mwh",
    "grid_imbalance","hour","dayofweek","power_mw_lag_1","power_mw_lag_24",
    "market_price_eur_mwh_lag_1","power_roll_6","price_roll_6"
]

def train():
    df = make_features(pd.read_parquet(SILVER/"energy_observations.parquet"))
    cutoff = df["timestamp"].quantile(.8)
    train_df = df[df.timestamp <= cutoff]
    test_df = df[df.timestamp > cutoff]

    model = XGBRegressor(
        n_estimators=300,max_depth=6,learning_rate=.05,
        subsample=.85,colsample_bytree=.85,
        objective="reg:squarederror",random_state=42
    )
    model.fit(train_df[FEATURES], train_df["power_mw"])
    pred = model.predict(test_df[FEATURES])

    mae = mean_absolute_error(test_df["power_mw"],pred)
    rmse = mean_squared_error(test_df["power_mw"],pred)**.5

    joblib.dump(model, MODELS/"generation_forecast.joblib")
    (MODELS/"metadata.json").write_text(json.dumps({
        "model":"XGBRegressor","features":FEATURES,
        "mae_mw":round(float(mae),3),"rmse_mw":round(float(rmse),3),
        "train_rows":len(train_df),"test_rows":len(test_df)
    },indent=2))

    out = test_df[["timestamp","asset_id","power_mw"]].copy()
    out["prediction_mw"] = pred
    out["absolute_error_mw"] = (out["power_mw"]-out["prediction_mw"]).abs()
    out.to_parquet(GOLD/"forecast_predictions.parquet",index=False)
    print(f"Model trained: MAE={mae:.2f} MW RMSE={rmse:.2f} MW")

if __name__ == "__main__":
    train()
