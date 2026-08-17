import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import SILVER, RAW, GOLD, MODELS
from .features import make_features


# =============================================================
# ASSET-SPECIFIC FEATURE DEFINITIONS
# =============================================================

SOLAR_FEATURES = [
    "temperature_forecast_c",
    "cloud_cover_forecast",
    "hour",
    "dayofweek",
    "power_mw_lag_1",
    "power_mw_lag_24",
    "power_roll_6",
    "market_price_eur_mwh",
    "grid_imbalance",
]


WIND_FEATURES = [
    "wind_speed_forecast_ms",
    "temperature_forecast_c",
    "hour",
    "dayofweek",
    "power_mw_lag_1",
    "power_mw_lag_24",
    "power_roll_6",
    "market_price_eur_mwh",
    "grid_imbalance",
]


BATTERY_FEATURES = [
    "state_of_charge",
    "market_price_eur_mwh",
    "grid_imbalance",
    "hour",
    "dayofweek",
    "power_mw_lag_1",
    "power_mw_lag_24",
    "power_roll_6",
    "price_roll_6",
]


ASSET_CONFIG = {
    "SOLAR_SE_01": {
        "features": SOLAR_FEATURES,
        "model_file": "solar_day_ahead.joblib",
    },
    "WIND_SE_01": {
        "features": WIND_FEATURES,
        "model_file": "wind_day_ahead.joblib",
    },
    "BATTERY_SE_01": {
        "features": BATTERY_FEATURES,
        "model_file": "battery_day_ahead.joblib",
    },
}


# =============================================================
# MODEL TRAINING
# =============================================================

def train_asset_model(
    df: pd.DataFrame,
    asset_id: str,
    features: list[str],
    model_file: str,
):
    print()
    print("=" * 60)
    print(f"Training asset: {asset_id}")
    print("=" * 60)

    asset_df = df[
        df["asset_id"] == asset_id
    ].copy()

    # ---------------------------------------------------------
    # Validate features
    # ---------------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in asset_df.columns
    ]

    if missing:
        raise ValueError(
            f"{asset_id} missing features: {missing}"
        )

    # ---------------------------------------------------------
    # Remove incomplete rows
    # ---------------------------------------------------------

    asset_df = asset_df.dropna(
        subset=features + ["target_power_mw_24h"]
    ).reset_index(drop=True)

    if len(asset_df) < 100:
        raise ValueError(
            f"Not enough training rows for {asset_id}: "
            f"{len(asset_df)}"
        )

    # ---------------------------------------------------------
    # Time-based split
    # ---------------------------------------------------------

    cutoff = asset_df["timestamp"].quantile(
        0.8
    )

    train_df = asset_df[
        asset_df["timestamp"] <= cutoff
    ].copy()

    test_df = asset_df[
        asset_df["timestamp"] > cutoff
    ].copy()

    # ---------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------

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
        train_df[features],
        train_df["target_power_mw_24h"],
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    predictions = model.predict(
        test_df[features]
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        test_df["target_power_mw_24h"],
        predictions,
    )

    rmse = mean_squared_error(
        test_df["target_power_mw_24h"],
        predictions,
    ) ** 0.5

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model_path = (
        MODELS / model_file
    )

    joblib.dump(
        model,
        model_path,
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    feature_importance = []

    for feature, importance in zip(
        features,
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
        key=lambda item: item["importance"],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

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
        / output["target_power_mw_24h"]
        .abs()
        .clip(lower=1)
        * 100
    )

    output_path = (
        GOLD
        / f"{asset_id.lower()}_day_ahead_predictions.parquet"
    )

    output.to_parquet(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print(
        f"Train rows : {len(train_df)}"
    )

    print(
        f"Test rows  : {len(test_df)}"
    )

    print(
        f"MAE        : {mae:.2f} MW"
    )

    print(
        f"RMSE       : {rmse:.2f} MW"
    )

    print()
    print("Feature importance:")

    for item in feature_importance:
        print(
            f"  {item['feature']}: "
            f"{item['importance']:.4f}"
        )

    return {
        "asset_id": asset_id,
        "model": "XGBRegressor",
        "forecast_horizon_hours": 24,
        "features": features,
        "mae_mw": round(float(mae), 3),
        "rmse_mw": round(float(rmse), 3),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "model_path": str(model_path),
        "prediction_path": str(output_path),
        "feature_importance": feature_importance,
    }


# =============================================================
# MAIN TRAINING PIPELINE
# =============================================================

def train():

    print()
    print("=" * 60)
    print("FLOWER ENERGY PLATFORM")
    print("Asset-Specific Day-Ahead Forecasting")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load telemetry
    # ---------------------------------------------------------

    telemetry = pd.read_parquet(
        SILVER / "energy_observations.parquet"
    )

    print(
        f"Telemetry rows: {len(telemetry)}"
    )

    # ---------------------------------------------------------
    # Shared feature engineering
    # ---------------------------------------------------------

    telemetry = make_features(
        telemetry
    )

    # ---------------------------------------------------------
    # Load weather forecasts
    # ---------------------------------------------------------

    weather = pd.read_csv(
        RAW / "weather_forecast.csv",
        parse_dates=["timestamp"],
    )

    print(
        f"Weather rows: {len(weather)}"
    )

    # ---------------------------------------------------------
    # Align weather forecasts
    #
    # Forecast issued at T
    # predicts weather at T+24.
    #
    # Therefore move the forecast timestamp backwards 24h.
    # ---------------------------------------------------------

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

    weather = weather.drop_duplicates(
        subset=[
            "timestamp",
            "asset_id",
        ]
    )

    # ---------------------------------------------------------
    # Merge telemetry + forecast weather
    # ---------------------------------------------------------

    df = telemetry.merge(
        weather,
        on=[
            "timestamp",
            "asset_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged rows: {len(df)}"
    )

    # ---------------------------------------------------------
    # Create future target
    # ---------------------------------------------------------

    df["target_power_mw_24h"] = (
        df.groupby("asset_id")["power_mw"]
        .shift(-24)
    )

    print(
        f"Rows before target cleanup: {len(df)}"
    )

    # ---------------------------------------------------------
    # Train each asset independently
    # ---------------------------------------------------------

    results = []

    for asset_id, config in ASSET_CONFIG.items():

        result = train_asset_model(
            df=df,
            asset_id=asset_id,
            features=config["features"],
            model_file=config["model_file"],
        )

        results.append(
            result
        )

    # ---------------------------------------------------------
    # Calculate overall weighted metrics
    # ---------------------------------------------------------

    total_test_rows = sum(
        result["test_rows"]
        for result in results
    )

    weighted_mae = sum(
        result["mae_mw"]
        * result["test_rows"]
        for result in results
    ) / total_test_rows

    weighted_rmse = sum(
        result["rmse_mw"]
        * result["test_rows"]
        for result in results
    ) / total_test_rows

    # ---------------------------------------------------------
    # Model registry
    # ---------------------------------------------------------

    registry = {
        "project": "Flower Energy Data Platform",
        "forecast_type": "asset_specific_day_ahead",
        "forecast_horizon_hours": 24,
        "models": results,
        "weighted_mae_mw": round(
            weighted_mae,
            3,
        ),
        "weighted_rmse_mw": round(
            weighted_rmse,
            3,
        ),
    }

    registry_path = (
        MODELS / "model_registry.json"
    )

    registry_path.write_text(
        json.dumps(
            registry,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL REGISTRY")
    print("=" * 60)

    for result in results:

        print(
            f"{result['asset_id']}: "
            f"MAE={result['mae_mw']:.2f} MW | "
            f"RMSE={result['rmse_mw']:.2f} MW"
        )

    print()
    print(
        f"Weighted MAE : "
        f"{weighted_mae:.2f} MW"
    )

    print(
        f"Weighted RMSE: "
        f"{weighted_rmse:.2f} MW"
    )

    print()
    print(
        f"Registry saved: {registry_path}"
    )


if __name__ == "__main__":
    train()