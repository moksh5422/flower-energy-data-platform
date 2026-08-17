import json

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import GOLD, MODELS


# ============================================================
# CONFIGURATION
# ============================================================

ASSETS = [
    "SOLAR_SE_01",
    "WIND_SE_01",
    "BATTERY_SE_01",
]


PREDICTION_FILES = {
    "SOLAR_SE_01": GOLD / "solar_se_01_day_ahead_predictions.parquet",
    "WIND_SE_01": GOLD / "wind_se_01_day_ahead_predictions.parquet",
    "BATTERY_SE_01": GOLD / "battery_se_01_day_ahead_predictions.parquet",
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(df: pd.DataFrame) -> dict:

    actual = df["target_power_mw_24h"]
    predicted = df["prediction_mw"]

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = mean_squared_error(
        actual,
        predicted,
    ) ** 0.5

    error = predicted - actual

    bias = error.mean()

    # MAPE can become unstable around zero.
    denominator = actual.abs().clip(lower=1)

    mape = (
        error.abs()
        / denominator
        * 100
    ).mean()

    return {
        "rows": int(len(df)),
        "mae_mw": round(float(mae), 3),
        "rmse_mw": round(float(rmse), 3),
        "bias_mw": round(float(bias), 3),
        "mape_pct": round(float(mape), 3),
        "mean_actual_mw": round(
            float(actual.mean()),
            3,
        ),
        "mean_prediction_mw": round(
            float(predicted.mean()),
            3,
        ),
    }


# ============================================================
# HOURLY PERFORMANCE
# ============================================================

def calculate_hourly_metrics(
    df: pd.DataFrame,
) -> list:

    df = df.copy()

    df["hour"] = (
        pd.to_datetime(
            df["timestamp"],
            utc=True,
        ).dt.hour
    )

    results = []

    for hour, group in df.groupby("hour"):

        if len(group) == 0:
            continue

        mae = mean_absolute_error(
            group["target_power_mw_24h"],
            group["prediction_mw"],
        )

        rmse = mean_squared_error(
            group["target_power_mw_24h"],
            group["prediction_mw"],
        ) ** 0.5

        results.append(
            {
                "hour": int(hour),
                "rows": int(len(group)),
                "mae_mw": round(
                    float(mae),
                    3,
                ),
                "rmse_mw": round(
                    float(rmse),
                    3,
                ),
            }
        )

    return results


# ============================================================
# WORST FORECASTS
# ============================================================

def calculate_worst_forecasts(
    df: pd.DataFrame,
    top_n: int = 10,
) -> list:

    result = df.copy()

    result["absolute_error_mw"] = (
        result["target_power_mw_24h"]
        - result["prediction_mw"]
    ).abs()

    result = result.sort_values(
        "absolute_error_mw",
        ascending=False,
    ).head(top_n)

    records = []

    for _, row in result.iterrows():

        records.append(
            {
                "timestamp": str(
                    row["timestamp"]
                ),
                "asset_id": row["asset_id"],
                "actual_mw": round(
                    float(
                        row[
                            "target_power_mw_24h"
                        ]
                    ),
                    3,
                ),
                "prediction_mw": round(
                    float(
                        row["prediction_mw"]
                    ),
                    3,
                ),
                "absolute_error_mw": round(
                    float(
                        row[
                            "absolute_error_mw"
                        ]
                    ),
                    3,
                ),
            }
        )

    return records


# ============================================================
# MODEL HEALTH
# ============================================================

def determine_model_health(
    mae: float,
    rmse: float,
) -> str:

    # These thresholds are intentionally simple
    # for the first monitoring version.

    if mae < 30:
        return "EXCELLENT"

    if mae < 60:
        return "GOOD"

    if mae < 100:
        return "WARNING"

    return "POOR"


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    print()
    print("=" * 60)
    print("FLOWER ENERGY PLATFORM")
    print("MODEL EVALUATION & MONITORING")
    print("=" * 60)

    all_predictions = []
    asset_results = {}

    # --------------------------------------------------------
    # Evaluate each asset
    # --------------------------------------------------------

    for asset_id in ASSETS:

        prediction_file = (
            PREDICTION_FILES[asset_id]
        )

        if not prediction_file.exists():

            print(
                f"WARNING: Missing prediction file "
                f"for {asset_id}"
            )

            continue

        df = pd.read_parquet(
            prediction_file
        )

        if df.empty:

            print(
                f"WARNING: Empty prediction file "
                f"for {asset_id}"
            )

            continue

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        )

        metrics = calculate_metrics(
            df
        )

        hourly = calculate_hourly_metrics(
            df
        )

        worst = calculate_worst_forecasts(
            df,
            top_n=10,
        )

        health = determine_model_health(
            metrics["mae_mw"],
            metrics["rmse_mw"],
        )

        asset_results[asset_id] = {
            "health": health,
            "metrics": metrics,
            "hourly_metrics": hourly,
            "worst_forecasts": worst,
        }

        all_predictions.append(
            df
        )

        print()
        print("-" * 60)
        print(asset_id)
        print("-" * 60)

        print(
            f"Health : {health}"
        )

        print(
            f"MAE    : "
            f"{metrics['mae_mw']:.2f} MW"
        )

        print(
            f"RMSE   : "
            f"{metrics['rmse_mw']:.2f} MW"
        )

        print(
            f"Bias   : "
            f"{metrics['bias_mw']:.2f} MW"
        )

        print(
            f"MAPE   : "
            f"{metrics['mape_pct']:.2f}%"
        )

    # --------------------------------------------------------
    # Combine predictions
    # --------------------------------------------------------

    if not all_predictions:

        raise RuntimeError(
            "No prediction files were found."
        )

    combined = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    overall_metrics = calculate_metrics(
        combined
    )

    overall_hourly = calculate_hourly_metrics(
        combined
    )

    overall_worst = calculate_worst_forecasts(
        combined,
        top_n=20,
    )

    # --------------------------------------------------------
    # Asset comparison
    # --------------------------------------------------------

    asset_summary = []

    for asset_id, result in asset_results.items():

        metrics = result["metrics"]

        asset_summary.append(
            {
                "asset_id": asset_id,
                "health": result["health"],
                "mae_mw": metrics["mae_mw"],
                "rmse_mw": metrics["rmse_mw"],
                "bias_mw": metrics["bias_mw"],
                "mape_pct": metrics["mape_pct"],
            }
        )

    # --------------------------------------------------------
    # Overall health
    # --------------------------------------------------------

    overall_health = determine_model_health(
        overall_metrics["mae_mw"],
        overall_metrics["rmse_mw"],
    )

    # --------------------------------------------------------
    # Monitoring report
    # --------------------------------------------------------

    report = {
        "project": "Flower Energy Data Platform",
        "evaluation_type": "day_ahead_forecast_monitoring",
        "forecast_horizon_hours": 24,

        "overall": {
            "health": overall_health,
            "metrics": overall_metrics,
        },

        "asset_summary": asset_summary,

        "asset_details": asset_results,

        "hourly_performance": overall_hourly,

        "worst_forecasts": overall_worst,

        "monitoring_notes": [
            "MAE measures average absolute forecast error.",
            "RMSE penalizes large forecast errors.",
            "Bias shows systematic over- or under-prediction.",
            "MAPE is calculated with a minimum denominator of 1 MW.",
            "Hourly metrics identify time periods with weaker forecasts.",
            "Worst forecasts identify the largest individual errors.",
        ],
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_path = (
        MODELS / "evaluation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Save combined evaluation dataset
    # --------------------------------------------------------

    evaluation_output = combined.copy()

    evaluation_output["error_mw"] = (
        evaluation_output["prediction_mw"]
        - evaluation_output[
            "target_power_mw_24h"
        ]
    )

    evaluation_output[
        "absolute_error_mw"
    ] = evaluation_output[
        "error_mw"
    ].abs()

    evaluation_output.to_parquet(
        GOLD / "forecast_evaluation.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Print final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("OVERALL MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Health : {overall_health}"
    )

    print(
        f"MAE    : "
        f"{overall_metrics['mae_mw']:.2f} MW"
    )

    print(
        f"RMSE   : "
        f"{overall_metrics['rmse_mw']:.2f} MW"
    )

    print(
        f"Bias   : "
        f"{overall_metrics['bias_mw']:.2f} MW"
    )

    print(
        f"MAPE   : "
        f"{overall_metrics['mape_pct']:.2f}%"
    )

    print()
    print("=" * 60)
    print("ASSET PERFORMANCE")
    print("=" * 60)

    for item in asset_summary:

        print(
            f"{item['asset_id']}: "
            f"MAE={item['mae_mw']:.2f} MW | "
            f"RMSE={item['rmse_mw']:.2f} MW | "
            f"Health={item['health']}"
        )

    print()
    print("=" * 60)
    print("WORST FORECASTS")
    print("=" * 60)

    for item in overall_worst[:10]:

        print(
            f"{item['timestamp']} | "
            f"{item['asset_id']} | "
            f"Actual={item['actual_mw']:.2f} MW | "
            f"Predicted={item['prediction_mw']:.2f} MW | "
            f"Error={item['absolute_error_mw']:.2f} MW"
        )

    print()
    print(
        f"Evaluation report saved: "
        f"{report_path}"
    )

    print(
        "Evaluation dataset saved: "
        f"{GOLD / 'forecast_evaluation.parquet'}"
    )


if __name__ == "__main__":
    evaluate()