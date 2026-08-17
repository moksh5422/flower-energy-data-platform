import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import SILVER, RAW, GOLD, MODELS
from .features import make_features


FEATURES = [
    "temperature_forecast_c",
    "wind_speed_forecast_ms",
    "cloud_cover_forecast",
    "market_price_eur_mwh",
    "grid_imbalance",
    "hour",
    "dayofweek",
    "power_mw_lag_1",
    "power_mw_lag_24",
    "power_roll_6",
    "price_roll_6",
]


def train():

    # =========================================================
    # 1. LOAD HISTORICAL TELEMETRY
    # =========================================================

    telemetry = pd.read_parquet(
        SILVER / "energy_observations.parquet"
    )

    telemetry = make_features(telemetry)

    print(
        f"Historical telemetry rows: {len(telemetry)}"
    )

    # =========================================================
    # 2. LOAD WEATHER FORECAST
    # =========================================================

    weather = pd.read_csv(
        RAW / "weather_forecast.csv",
        parse_dates=["timestamp"],
    )

    print(
        f"Weather forecast rows: {len(weather)}"
    )

    # =========================================================
    # 3. ALIGN WEATHER FORECAST TO 24-HOUR HORIZON
    # =========================================================
    #
    # The weather forecast at timestamp T represents
    # the weather expected for T.
    #
    # We shift it backwards by 24 hours:
    #
    # Forecast generated at:
    #     T
    #
    # Forecast is for:
    #     T + 24h
    #
    # Therefore the model row at T gets weather forecast
    # information for T + 24h.
    # =========================================================

    weather["forecast_for_timestamp"] = (
        weather["timestamp"]
        - pd.Timedelta(hours=24)
    )

    weather = weather.rename(
        columns={
            "timestamp": "forecast_issue_timestamp",
            "forecast_for_timestamp": "timestamp",
        }
    )

    weather = weather[
        [
            "timestamp",
            "asset_id",
            "temperature_forecast_c",
            "wind_speed_forecast_ms",
            "cloud_cover_forecast",
        ]
    ].copy()

    # Make absolutely sure merge keys are unique.
    weather = weather.drop_duplicates(
        subset=["timestamp", "asset_id"]
    )

    # =========================================================
    # 4. MERGE HISTORICAL FEATURES + WEATHER FORECAST
    # =========================================================

    df = telemetry.merge(
        weather,
        on=["timestamp", "asset_id"],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged modeling rows: {len(df)}"
    )

    # =========================================================
    # 5. CREATE 24-HOUR FUTURE TARGET
    # =========================================================

    df["target_power_mw_24h"] = (
        df.groupby("asset_id")["power_mw"]
        .shift(-24)
    )

    # =========================================================
    # 6. REMOVE INCOMPLETE ROWS
    # =========================================================

    df = df.dropna(
        subset=FEATURES + ["target_power_mw_24h"]
    ).reset_index(drop=True)

    print(
        f"Final modeling rows: {len(df)}"
    )

    # =========================================================
    # 7. TIME-BASED TRAIN / TEST SPLIT
    # =========================================================

    cutoff = df["timestamp"].quantile(0.8)

    train_df = df[
        df["timestamp"] <= cutoff
    ].copy()

    test_df = df[
        df["timestamp"] > cutoff
    ].copy()

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Testing rows: {len(test_df)}"
    )

    # =========================================================
    # 8. TRAIN XGBOOST
    # =========================================================

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )

    model.fit(
        train_df[FEATURES],
        train_df["target_power_mw_24h"],
    )

    # =========================================================
    # 9. PREDICT
    # =========================================================

    predictions = model.predict(
        test_df[FEATURES]
    )

    # =========================================================
    # 10. EVALUATION
    # =========================================================

    mae = mean_absolute_error(
        test_df["target_power_mw_24h"],
        predictions,
    )

    rmse = mean_squared_error(
        test_df["target_power_mw_24h"],
        predictions,
    ) ** 0.5

    print()
    print(
        "=============================================="
    )
    print(
        "Weather-Aware Day-Ahead Forecast"
    )
    print(
        "=============================================="
    )
    print(
        f"MAE  : {mae:.2f} MW"
    )
    print(
        f"RMSE : {rmse:.2f} MW"
    )
    print(
        "=============================================="
    )

    # =========================================================
    # 11. SAVE MODEL
    # =========================================================

    model_path = (
        MODELS
        / "day_ahead_weather_forecast.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    # =========================================================
    # 12. FEATURE IMPORTANCE
    # =========================================================

    feature_importance = []

    for feature, importance in zip(
        FEATURES,
        model.feature_importances_,
    ):
        feature_importance.append(
            {
                "feature": feature,
                "importance": round(
                    float(importance),
                    5,
                ),
            }
        )

    feature_importance.sort(
        key=lambda x: x["importance"],
        reverse=True,
    )

    # =========================================================
    # 13. SAVE METADATA
    # =========================================================

    metadata = {
        "model": "XGBRegressor",
        "forecast_type": "day_ahead",
        "forecast_horizon_hours": 24,
        "weather_forecast": True,
        "features": FEATURES,
        "mae_mw": round(
            float(mae),
            3,
        ),
        "rmse_mw": round(
            float(rmse),
            3,
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "model_path": str(
            model_path
        ),
        "feature_importance": feature_importance,
    }

    metadata_path = (
        MODELS
        / "day_ahead_weather_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # =========================================================
    # 14. SAVE PREDICTIONS
    # =========================================================

    output = test_df[
        [
            "timestamp",
            "asset_id",
            "power_mw",
            "target_power_mw_24h",
        ]
    ].copy()

    output["prediction_mw"] = predictions

    output["absolute_error_mw"] = (
        output["target_power_mw_24h"]
        - output["prediction_mw"]
    ).abs()

    output["error_pct"] = (
        output["absolute_error_mw"]
        / output["target_power_mw_24h"].abs().clip(
            lower=1
        )
        * 100
    )

    output.to_parquet(
        GOLD
        / "day_ahead_weather_predictions.parquet",
        index=False,
    )

    # =========================================================
    # 15. ASSET-LEVEL EVALUATION
    # =========================================================

    print()
    print(
        "Asset-level performance:"
    )

    for asset_id, group in output.groupby(
        "asset_id"
    ):

        asset_mae = mean_absolute_error(
            group["target_power_mw_24h"],
            group["prediction_mw"],
        )

        asset_rmse = mean_squared_error(
            group["target_power_mw_24h"],
            group["prediction_mw"],
        ) ** 0.5

        print(
            f"{asset_id}: "
            f"MAE={asset_mae:.2f} MW "
            f"RMSE={asset_rmse:.2f} MW"
        )

    # =========================================================
    # 16. TOP FEATURES
    # =========================================================

    print()
    print(
        "Top feature importance:"
    )

    for item in feature_importance:
        print(
            f"{item['feature']}: "
            f"{item['importance']:.4f}"
        )


if __name__ == "__main__":
    train()