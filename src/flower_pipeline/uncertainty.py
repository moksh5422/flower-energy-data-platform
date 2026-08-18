import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import GOLD, MODELS


ASSET = "WIND_SE_01"

INPUT_DATA = GOLD / "wind_residual_predictions.parquet"
OUTPUT_DATA = GOLD / "wind_uncertainty_predictions.parquet"
METADATA = MODELS / "wind_uncertainty_metadata.json"


def conformal_quantile(errors, alpha=0.20):
    """
    Calculate a finite-sample conformal quantile
    from absolute calibration errors.
    """

    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        raise ValueError(
            "No valid calibration errors available."
        )

    errors = np.sort(errors)

    n = len(errors)

    rank = int(
        np.ceil((n + 1) * (1 - alpha))
    ) - 1

    rank = min(
        max(rank, 0),
        n - 1,
    )

    return float(errors[rank])


def train():

    print("=" * 60)
    print("FLOWER ENERGY PLATFORM")
    print("V7 UNCERTAINTY-AWARE WIND FORECAST")
    print("=" * 60)

    if not INPUT_DATA.exists():
        raise FileNotFoundError(
            f"V6 prediction file not found: {INPUT_DATA}"
        )

    df = pd.read_parquet(INPUT_DATA).copy()

    # ---------------------------------------------------------
    # Filter wind asset
    # ---------------------------------------------------------

    df = df[
        df["asset_id"] == ASSET
    ].copy()

    # ---------------------------------------------------------
    # Validate V6 schema
    # ---------------------------------------------------------

    required_columns = [
        "timestamp",
        "asset_id",
        "target_power_mw",
        "final_prediction_mw",
        "absolute_error_mw",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"V6 rows: {len(df)}")

    # ---------------------------------------------------------
    # Time-based calibration / test split
    # ---------------------------------------------------------

    cutoff = df["timestamp"].quantile(0.80)

    calibration = df[
        df["timestamp"] <= cutoff
    ].copy()

    test = df[
        df["timestamp"] > cutoff
    ].copy()

    print(
        f"Calibration rows: {len(calibration)}"
    )

    print(
        f"Testing rows    : {len(test)}"
    )

    if calibration.empty:
        raise ValueError(
            "Calibration dataset is empty."
        )

    if test.empty:
        raise ValueError(
            "Testing dataset is empty."
        )

    # ---------------------------------------------------------
    # Calibration errors
    # ---------------------------------------------------------

    calibration_errors = (
        calibration["target_power_mw"]
        - calibration["final_prediction_mw"]
    ).abs()

    # ---------------------------------------------------------
    # 80% prediction interval
    #
    # alpha = 0.20
    # ---------------------------------------------------------

    alpha = 0.20

    conformal_width = conformal_quantile(
        calibration_errors,
        alpha,
    )

    # ---------------------------------------------------------
    # Point forecast
    # ---------------------------------------------------------

    test["point_forecast_mw"] = (
        test["final_prediction_mw"]
    )

    # ---------------------------------------------------------
    # Prediction interval
    # ---------------------------------------------------------

    test["lower_10_mw"] = (
        test["point_forecast_mw"]
        - conformal_width
    )

    test["upper_90_mw"] = (
        test["point_forecast_mw"]
        + conformal_width
    )

    # Generation cannot be negative.
    test["lower_10_mw"] = (
        test["lower_10_mw"]
        .clip(lower=0)
    )

    # Wind capacity is 1100 MW in our generated dataset.
    WIND_CAPACITY_MW = 1100.0

    test["upper_90_mw"] = (
        test["upper_90_mw"]
        .clip(
            upper=WIND_CAPACITY_MW
        )
    )

    # ---------------------------------------------------------
    # Interval width
    # ---------------------------------------------------------

    test["interval_width_mw"] = (
        test["upper_90_mw"]
        - test["lower_10_mw"]
    )

    # ---------------------------------------------------------
    # Forecast error
    # ---------------------------------------------------------

    test["absolute_error_mw"] = (
        test["target_power_mw"]
        - test["point_forecast_mw"]
    ).abs()

    # ---------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------

    test["covered"] = (
        (
            test["target_power_mw"]
            >= test["lower_10_mw"]
        )
        &
        (
            test["target_power_mw"]
            <= test["upper_90_mw"]
        )
    )

    # ---------------------------------------------------------
    # Point forecast metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        test["target_power_mw"],
        test["point_forecast_mw"],
    )

    rmse = mean_squared_error(
        test["target_power_mw"],
        test["point_forecast_mw"],
    ) ** 0.5

    coverage = (
        test["covered"].mean()
        * 100
    )

    mean_interval_width = (
        test["interval_width_mw"]
        .mean()
    )

    # ---------------------------------------------------------
    # Bias
    # ---------------------------------------------------------

    bias = (
        test["point_forecast_mw"]
        - test["target_power_mw"]
    ).mean()

    # ---------------------------------------------------------
    # Worst forecasts
    # ---------------------------------------------------------

    worst = (
        test[
            [
                "timestamp",
                "asset_id",
                "target_power_mw",
                "point_forecast_mw",
                "lower_10_mw",
                "upper_90_mw",
                "absolute_error_mw",
                "covered",
            ]
        ]
        .sort_values(
            "absolute_error_mw",
            ascending=False,
        )
        .head(10)
    )

    worst_forecasts = []

    for _, row in worst.iterrows():

        worst_forecasts.append(
            {
                "timestamp": str(
                    row["timestamp"]
                ),
                "asset_id": row["asset_id"],
                "actual_mw": round(
                    float(
                        row["target_power_mw"]
                    ),
                    3,
                ),
                "forecast_mw": round(
                    float(
                        row["point_forecast_mw"]
                    ),
                    3,
                ),
                "lower_10_mw": round(
                    float(
                        row["lower_10_mw"]
                    ),
                    3,
                ),
                "upper_90_mw": round(
                    float(
                        row["upper_90_mw"]
                    ),
                    3,
                ),
                "absolute_error_mw": round(
                    float(
                        row["absolute_error_mw"]
                    ),
                    3,
                ),
                "covered": bool(
                    row["covered"]
                ),
            }
        )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    output_columns = [
        "timestamp",
        "asset_id",
        "target_power_mw",
        "point_forecast_mw",
        "lower_10_mw",
        "upper_90_mw",
        "interval_width_mw",
        "absolute_error_mw",
        "covered",
    ]

    test[
        output_columns
    ].to_parquet(
        OUTPUT_DATA,
        index=False,
    )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = {

        "model":
            "V6 Wind Residual Calibration",

        "uncertainty_method":
            "Conformal Prediction",

        "asset_id":
            ASSET,

        "coverage_target_percent":
            80.0,

        "empirical_coverage_percent":
            round(
                float(coverage),
                3,
            ),

        "conformal_interval_half_width_mw":
            round(
                float(conformal_width),
                3,
            ),

        "mean_interval_width_mw":
            round(
                float(mean_interval_width),
                3,
            ),

        "mae_mw":
            round(
                float(mae),
                3,
            ),

        "rmse_mw":
            round(
                float(rmse),
                3,
            ),

        "bias_mw":
            round(
                float(bias),
                3,
            ),

        "calibration_rows":
            len(calibration),

        "test_rows":
            len(test),

        "worst_forecasts":
            worst_forecasts,
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("V7 RESULTS")
    print("=" * 60)

    print(
        f"Point Forecast MAE : "
        f"{mae:.2f} MW"
    )

    print(
        f"Point Forecast RMSE: "
        f"{rmse:.2f} MW"
    )

    print(
        f"Bias               : "
        f"{bias:.2f} MW"
    )

    print(
        f"Target Coverage    : "
        f"80.00%"
    )

    print(
        f"Actual Coverage    : "
        f"{coverage:.2f}%"
    )

    print(
        f"Conformal Width    : "
        f"±{conformal_width:.2f} MW"
    )

    print(
        f"Mean Interval Width: "
        f"{mean_interval_width:.2f} MW"
    )

    print()
    print("=" * 60)
    print("WORST FORECASTS")
    print("=" * 60)

    for item in worst_forecasts:

        print(
            f"{item['timestamp']} | "
            f"Actual={item['actual_mw']:.2f} MW | "
            f"Forecast={item['forecast_mw']:.2f} MW | "
            f"Interval=["
            f"{item['lower_10_mw']:.2f}, "
            f"{item['upper_90_mw']:.2f}] | "
            f"Error={item['absolute_error_mw']:.2f} MW | "
            f"Covered={item['covered']}"
        )

    print()
    print(
        f"Predictions saved: "
        f"{OUTPUT_DATA}"
    )

    print(
        f"Metadata saved: "
        f"{METADATA}"
    )


if __name__ == "__main__":
    train()