import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import SILVER, RAW, GOLD, MODELS
from .features import make_features


WIND_CAPACITY_MW = 1100.0


# ============================================================
# PHYSICS-BASED WIND TURBINE POWER CURVE
# ============================================================

def wind_power_curve(wind_speed):
    """
    Simplified turbine power curve.

    < 3 m/s   -> no generation
    3-12 m/s   -> cubic power relationship
    12-25 m/s  -> rated generation
    > 25 m/s   -> cut-out
    """

    wind_speed = np.asarray(
        wind_speed,
        dtype=float,
    )

    power_factor = np.zeros_like(
        wind_speed
    )

    # Cubic operating region
    mask_cubic = (
        (wind_speed >= 3.0)
        & (wind_speed < 12.0)
    )

    power_factor[mask_cubic] = (
        (wind_speed[mask_cubic] - 3.0)
        / 9.0
    ) ** 3

    # Rated operating region
    mask_rated = (
        (wind_speed >= 12.0)
        & (wind_speed <= 25.0)
    )

    power_factor[mask_rated] = 1.0

    # Outside operating range remains 0

    return np.clip(
        power_factor,
        0,
        1,
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_wind_features():

    telemetry = pd.read_parquet(
        SILVER / "energy_observations.parquet"
    )

    telemetry = make_features(
        telemetry
    )

    telemetry = telemetry[
        telemetry["asset_id"]
        == "WIND_SE_01"
    ].copy()

    weather = pd.read_csv(
        RAW / "weather_forecast.csv",
        parse_dates=["timestamp"],
    )

    # Forecast issued at T predicts T+24.
    weather["forecast_for_timestamp"] = (
        weather["timestamp"]
        - pd.Timedelta(hours=24)
    )

    weather = weather.rename(
        columns={
            "timestamp":
                "forecast_issue_timestamp",
            "forecast_for_timestamp":
                "timestamp",
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

    weather = weather[
        weather["asset_id"]
        == "WIND_SE_01"
    ]

    weather = weather.drop_duplicates(
        subset=[
            "timestamp",
            "asset_id",
        ]
    )

    df = telemetry.merge(
        weather,
        on=[
            "timestamp",
            "asset_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Future target
    # --------------------------------------------------------

    df["target_power_mw_24h"] = (
        df["power_mw"]
        .shift(-24)
    )

    # --------------------------------------------------------
    # Physics features
    # --------------------------------------------------------

    df["wind_power_curve_factor"] = (
        wind_power_curve(
            df["wind_speed_forecast_ms"]
        )
    )

    df["physics_wind_power_mw"] = (
        WIND_CAPACITY_MW
        * df["wind_power_curve_factor"]
    )

    # Capacity-normalized historical generation
    df["historical_capacity_factor"] = (
        df["power_mw_lag_1"]
        / WIND_CAPACITY_MW
    )

    df["historical_capacity_factor_24h"] = (
        df["power_mw_lag_24"]
        / WIND_CAPACITY_MW
    )

    features = [
        "wind_speed_forecast_ms",
        "temperature_forecast_c",
        "cloud_cover_forecast",
        "wind_power_curve_factor",
        "physics_wind_power_mw",
        "historical_capacity_factor",
        "historical_capacity_factor_24h",
        "power_roll_6",
        "hour",
        "dayofweek",
        "market_price_eur_mwh",
        "grid_imbalance",
    ]

    df = df.dropna(
        subset=features
        + ["target_power_mw_24h"]
    ).reset_index(drop=True)

    return df, features


# ============================================================
# TRAIN V5
# ============================================================

def train():

    print()
    print("=" * 60)
    print("FLOWER ENERGY PLATFORM")
    print("V5 PHYSICS-INFORMED WIND FORECAST")
    print("=" * 60)

    df, features = build_wind_features()

    print(
        f"Modeling rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Time-based split
    # --------------------------------------------------------

    cutoff = df["timestamp"].quantile(
        0.8
    )

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
        f"Testing rows : {len(test_df)}"
    )

    # --------------------------------------------------------
    # XGBoost correction model
    # --------------------------------------------------------

    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )

    model.fit(
        train_df[features],
        train_df["target_power_mw_24h"],
    )

    predictions = model.predict(
        test_df[features]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        test_df["target_power_mw_24h"],
        predictions,
    )

    rmse = mean_squared_error(
        test_df["target_power_mw_24h"],
        predictions,
    ) ** 0.5

    physics_mae = mean_absolute_error(
        test_df["target_power_mw_24h"],
        test_df["physics_wind_power_mw"],
    )

    physics_rmse = mean_squared_error(
        test_df["target_power_mw_24h"],
        test_df["physics_wind_power_mw"],
    ) ** 0.5

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODELS
        / "wind_physics_day_ahead.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    output = test_df[
        [
            "timestamp",
            "asset_id",
            "wind_speed_forecast_ms",
            "wind_power_curve_factor",
            "physics_wind_power_mw",
            "power_mw",
            "target_power_mw_24h",
        ]
    ].copy()

    output["prediction_mw"] = predictions

    output["absolute_error_mw"] = (
        output["target_power_mw_24h"]
        - output["prediction_mw"]
    ).abs()

    output["physics_absolute_error_mw"] = (
        output["target_power_mw_24h"]
        - output["physics_wind_power_mw"]
    ).abs()

    prediction_path = (
        GOLD
        / "wind_physics_predictions.parquet"
    )

    output.to_parquet(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = []

    for feature, value in zip(
        features,
        model.feature_importances_,
    ):

        importance.append(
            {
                "feature": feature,
                "importance": round(
                    float(value),
                    5,
                ),
            }
        )

    importance.sort(
        key=lambda x: x["importance"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model": "XGBRegressor",
        "version": "V5",
        "type": "physics_informed_wind_forecast",
        "forecast_horizon_hours": 24,
        "asset_id": "WIND_SE_01",
        "capacity_mw": WIND_CAPACITY_MW,
        "features": features,
        "model_mae_mw": round(
            float(mae),
            3,
        ),
        "model_rmse_mw": round(
            float(rmse),
            3,
        ),
        "physics_only_mae_mw": round(
            float(physics_mae),
            3,
        ),
        "physics_only_rmse_mw": round(
            float(physics_rmse),
            3,
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "model_path": str(model_path),
        "prediction_path": str(
            prediction_path
        ),
        "feature_importance": importance,
    }

    metadata_path = (
        MODELS
        / "wind_physics_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V5 RESULTS")
    print("=" * 60)

    print(
        f"Physics-only MAE : "
        f"{physics_mae:.2f} MW"
    )

    print(
        f"Physics-only RMSE: "
        f"{physics_rmse:.2f} MW"
    )

    print()

    print(
        f"V5 Model MAE     : "
        f"{mae:.2f} MW"
    )

    print(
        f"V5 Model RMSE    : "
        f"{rmse:.2f} MW"
    )

    print()
    print("Feature importance:")

    for item in importance:

        print(
            f"  {item['feature']}: "
            f"{item['importance']:.4f}"
        )

    print()
    print(
        f"Model saved: {model_path}"
    )

    print(
        f"Predictions saved: "
        f"{prediction_path}"
    )

    print(
        f"Metadata saved: "
        f"{metadata_path}"
    )


if __name__ == "__main__":
    train()