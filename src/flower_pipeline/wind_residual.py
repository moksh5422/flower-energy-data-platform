import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import SILVER, RAW, GOLD, MODELS
from .features import make_features


WIND_ASSET = "WIND_SE_01"
WIND_CAPACITY_MW = 1100.0


def wind_power_curve(wind_speed):
    """Simplified wind turbine power curve."""

    wind_speed = np.asarray(
        wind_speed,
        dtype=float,
    )

    factor = np.zeros_like(wind_speed)

    # Cut-in -> rated
    mask = (
        (wind_speed >= 3.0)
        & (wind_speed < 12.0)
    )

    factor[mask] = (
        (wind_speed[mask] - 3.0) / 9.0
    ) ** 3

    # Rated -> cut-out
    mask_rated = (
        (wind_speed >= 12.0)
        & (wind_speed <= 25.0)
    )

    factor[mask_rated] = 1.0

    return np.clip(
        factor,
        0,
        1,
    )


def build_dataset():

    # ========================================================
    # HISTORICAL TELEMETRY
    # ========================================================

    telemetry = pd.read_parquet(
        SILVER / "energy_observations.parquet"
    )

    telemetry = make_features(
        telemetry
    )

    telemetry = telemetry[
        telemetry["asset_id"] == WIND_ASSET
    ].copy()

    # ========================================================
    # WEATHER FORECAST
    # ========================================================

    weather = pd.read_csv(
        RAW / "weather_forecast.csv",
        parse_dates=["timestamp"],
    )

    weather = weather[
        weather["asset_id"] == WIND_ASSET
    ].copy()

    # IMPORTANT:
    #
    # The weather timestamp represents the forecasted hour.
    # Therefore we DO NOT shift the timestamp here.
    #
    # The 24-hour prediction target is created from telemetry
    # separately using shift(-24).

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

    # ========================================================
    # MERGE
    # ========================================================

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
        f"Telemetry rows: {len(telemetry)}"
    )

    print(
        f"Weather rows: {len(weather)}"
    )

    print(
        f"Merged rows: {len(df)}"
    )

    # ========================================================
    # 24-HOUR FUTURE TARGET
    # ========================================================

    #
    # At timestamp T:
    #
    # features = weather forecast at T
    #
    # target = actual generation at T+24h
    #
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    df["target_power_mw"] = (
        df["power_mw"]
        .shift(-24)
    )

    # ========================================================
    # PHYSICS FORECAST
    # ========================================================

    df["wind_power_curve_factor"] = (
        wind_power_curve(
            df["wind_speed_forecast_ms"]
        )
    )

    df["physics_forecast_mw"] = (
        WIND_CAPACITY_MW
        * df["wind_power_curve_factor"]
    )

    # ========================================================
    # RESIDUAL
    # ========================================================

    df["physics_residual_mw"] = (
        df["target_power_mw"]
        - df["physics_forecast_mw"]
    )

    # ========================================================
    # PHYSICS FEATURES
    # ========================================================

    df["wind_speed_squared"] = (
        df["wind_speed_forecast_ms"] ** 2
    )

    df["wind_speed_cubed"] = (
        df["wind_speed_forecast_ms"] ** 3
    )

    df["physics_capacity_factor"] = (
        df["physics_forecast_mw"]
        / WIND_CAPACITY_MW
    )

    df["historical_capacity_factor"] = (
        df["power_mw_lag_1"]
        / WIND_CAPACITY_MW
    )

    df["historical_capacity_factor_24h"] = (
        df["power_mw_lag_24"]
        / WIND_CAPACITY_MW
    )

    features = [
        "physics_forecast_mw",
        "wind_power_curve_factor",
        "wind_speed_forecast_ms",
        "wind_speed_squared",
        "wind_speed_cubed",
        "temperature_forecast_c",
        "cloud_cover_forecast",
        "historical_capacity_factor",
        "historical_capacity_factor_24h",
        "power_roll_6",
        "hour",
        "dayofweek",
        "market_price_eur_mwh",
        "grid_imbalance",
    ]

    df = df.dropna(
        subset=features + [
            "target_power_mw",
            "physics_residual_mw",
        ]
    ).reset_index(drop=True)

    return df, features


def train():

    print()
    print("=" * 60)
    print("FLOWER ENERGY PLATFORM")
    print("V6 WIND RESIDUAL CALIBRATION")
    print("=" * 60)

    df, features = build_dataset()

    print(
        f"Final modeling rows: {len(df)}"
    )

    # ========================================================
    # TIME SPLIT
    # ========================================================

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

    # ========================================================
    # PHYSICS BASELINE
    # ========================================================

    physics_mae = mean_absolute_error(
        test_df["target_power_mw"],
        test_df["physics_forecast_mw"],
    )

    physics_rmse = mean_squared_error(
        test_df["target_power_mw"],
        test_df["physics_forecast_mw"],
    ) ** 0.5

    # ========================================================
    # RESIDUAL MODEL
    # ========================================================

    model = XGBRegressor(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_alpha=0.2,
        reg_lambda=1.5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )

    model.fit(
        train_df[features],
        train_df["physics_residual_mw"],
    )

    correction = model.predict(
        test_df[features]
    )

    correction = np.clip(
        correction,
        -WIND_CAPACITY_MW,
        WIND_CAPACITY_MW,
    )

    # ========================================================
    # FINAL FORECAST
    # ========================================================

    final_prediction = (
        test_df["physics_forecast_mw"].to_numpy()
        + correction
    )

    final_prediction = np.clip(
        final_prediction,
        0,
        WIND_CAPACITY_MW,
    )

    # ========================================================
    # V6 METRICS
    # ========================================================

    mae = mean_absolute_error(
        test_df["target_power_mw"],
        final_prediction,
    )

    rmse = mean_squared_error(
        test_df["target_power_mw"],
        final_prediction,
    ) ** 0.5

    mae_improvement = (
        physics_mae - mae
    )

    rmse_improvement = (
        physics_rmse - rmse
    )

    mae_improvement_pct = (
        mae_improvement
        / physics_mae
        * 100
    )

    rmse_improvement_pct = (
        rmse_improvement
        / physics_rmse
        * 100
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODELS
        / "wind_residual_calibration.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    output = test_df[
        [
            "timestamp",
            "asset_id",
            "wind_speed_forecast_ms",
            "physics_forecast_mw",
            "target_power_mw",
        ]
    ].copy()

    output["predicted_correction_mw"] = (
        correction
    )

    output["final_prediction_mw"] = (
        final_prediction
    )

    output["physics_error_mw"] = (
        output["target_power_mw"]
        - output["physics_forecast_mw"]
    )

    output["final_error_mw"] = (
        output["target_power_mw"]
        - output["final_prediction_mw"]
    )

    output["absolute_error_mw"] = (
        output["final_error_mw"].abs()
    )

    prediction_path = (
        GOLD
        / "wind_residual_predictions.parquet"
    )

    output.to_parquet(
        prediction_path,
        index=False,
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

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

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "model": "XGBRegressor",
        "version": "V6",
        "type": "wind_residual_calibration",
        "forecast_horizon_hours": 24,
        "asset_id": WIND_ASSET,
        "capacity_mw": WIND_CAPACITY_MW,

        "physics_baseline": {
            "mae_mw": round(
                float(physics_mae),
                3,
            ),
            "rmse_mw": round(
                float(physics_rmse),
                3,
            ),
        },

        "v6_model": {
            "mae_mw": round(
                float(mae),
                3,
            ),
            "rmse_mw": round(
                float(rmse),
                3,
            ),
        },

        "improvement": {
            "mae_mw": round(
                float(mae_improvement),
                3,
            ),
            "rmse_mw": round(
                float(rmse_improvement),
                3,
            ),
            "mae_percent": round(
                float(mae_improvement_pct),
                2,
            ),
            "rmse_percent": round(
                float(rmse_improvement_pct),
                2,
            ),
        },

        "features": features,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_importance": importance,
    }

    metadata_path = (
        MODELS
        / "wind_residual_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("V6 RESULTS")
    print("=" * 60)

    print(
        f"Physics MAE : "
        f"{physics_mae:.2f} MW"
    )

    print(
        f"Physics RMSE: "
        f"{physics_rmse:.2f} MW"
    )

    print()

    print(
        f"V6 MAE      : "
        f"{mae:.2f} MW"
    )

    print(
        f"V6 RMSE     : "
        f"{rmse:.2f} MW"
    )

    print()

    print(
        f"MAE improvement : "
        f"{mae_improvement:.2f} MW "
        f"({mae_improvement_pct:.2f}%)"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_improvement:.2f} MW "
        f"({rmse_improvement_pct:.2f}%)"
    )

    print()
    print("Top features:")

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