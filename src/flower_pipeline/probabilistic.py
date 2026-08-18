import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GOLD, MODELS


ASSET_ID = "WIND_SE_01"
TARGET_COVERAGE = 0.80


def load_v8_predictions():
    path = GOLD / "wind_calibrated_predictions.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"V8 predictions not found: {path}"
        )

    df = pd.read_parquet(path)

    required = [
        "timestamp",
        "asset_id",
        "wind_speed_forecast_ms",
        "final_prediction_mw",
        "target_power_mw",
        "absolute_error_mw",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def build_probabilistic_forecast(df):
    x = df.copy()

    # ---------------------------------------------------------
    # Estimate uncertainty from V8 forecast errors
    # ---------------------------------------------------------

    errors = (
        x["target_power_mw"]
        - x["final_prediction_mw"]
    )

    abs_errors = errors.abs()

    # Robust uncertainty estimate.
    # MAD is less sensitive to extreme wind events.
    median_error = errors.median()

    mad = np.median(
        np.abs(errors - median_error)
    )

    robust_sigma = 1.4826 * mad

    # Fall back to empirical standard deviation
    # if MAD becomes too small.
    if robust_sigma < 1:
        robust_sigma = errors.std()

    # ---------------------------------------------------------
    # Probabilistic quantiles
    # ---------------------------------------------------------

    # Normal approximation around V8 point forecast.
    #
    # P10 ≈ mean - 1.2816 sigma
    # P50 = mean
    # P90 ≈ mean + 1.2816 sigma

    z90 = 1.2815515655

    x["p50_mw"] = x["final_prediction_mw"]

    x["p10_mw"] = (
        x["p50_mw"]
        - z90 * robust_sigma
    )

    x["p90_mw"] = (
        x["p50_mw"]
        + z90 * robust_sigma
    )

    # Wind farm physical limits.
    capacity_mw = 1100.0

    x["p10_mw"] = x["p10_mw"].clip(
        0,
        capacity_mw,
    )

    x["p50_mw"] = x["p50_mw"].clip(
        0,
        capacity_mw,
    )

    x["p90_mw"] = x["p90_mw"].clip(
        0,
        capacity_mw,
    )

    # ---------------------------------------------------------
    # Risk metrics
    # ---------------------------------------------------------

    x["interval_width_mw"] = (
        x["p90_mw"]
        - x["p10_mw"]
    )

    x["downside_risk_mw"] = (
        x["p50_mw"]
        - x["p10_mw"]
    )

    x["upside_potential_mw"] = (
        x["p90_mw"]
        - x["p50_mw"]
    )

    # Probability that generation is below 25%
    # of installed capacity.
    low_generation_threshold = (
        0.25 * capacity_mw
    )

    # Normal CDF approximation.
    from math import erf, sqrt

    sigma = max(robust_sigma, 1.0)

    z = (
        low_generation_threshold
        - x["p50_mw"]
    ) / sigma

    x["prob_low_generation"] = (
        0.5
        * (
            1
            + np.vectorize(erf)(
                z / sqrt(2)
            )
        )
    )

    x["prob_low_generation"] = (
        x["prob_low_generation"]
        .clip(0, 1)
    )

    # ---------------------------------------------------------
    # Confidence score
    # ---------------------------------------------------------

    # Smaller uncertainty = higher confidence.
    #
    # 1.0 = very narrow uncertainty
    # 0.0 = very wide uncertainty

    normalized_width = (
        x["interval_width_mw"]
        / capacity_mw
    )

    x["forecast_confidence"] = (
        1
        - normalized_width
    ).clip(0, 1)

    # ---------------------------------------------------------
    # Risk classification
    # ---------------------------------------------------------

    conditions = [
        x["prob_low_generation"] >= 0.75,
        x["prob_low_generation"] >= 0.50,
        x["forecast_confidence"] < 0.30,
        x["forecast_confidence"] < 0.50,
    ]

    choices = [
        "CRITICAL_LOW_GENERATION",
        "HIGH_LOW_GENERATION_RISK",
        "LOW_CONFIDENCE",
        "MEDIUM_CONFIDENCE",
    ]

    x["risk_class"] = np.select(
        conditions,
        choices,
        default="NORMAL",
    )

    # ---------------------------------------------------------
    # Dispatch recommendation
    # ---------------------------------------------------------

    x["dispatch_recommendation"] = np.select(
        [
            x["risk_class"]
            == "CRITICAL_LOW_GENERATION",

            x["risk_class"]
            == "HIGH_LOW_GENERATION_RISK",

            x["risk_class"]
            == "LOW_CONFIDENCE",

            x["risk_class"]
            == "MEDIUM_CONFIDENCE",
        ],
        [
            "ACTIVATE_RESERVE",
            "PREPARE_RESERVE",
            "REDUCE_CONFIDENCE_IN_COMMITMENT",
            "MONITOR",
        ],
        default="NORMAL_DISPATCH",
    )

    # ---------------------------------------------------------
    # Actual coverage
    # ---------------------------------------------------------

    x["interval_covered"] = (
        (x["target_power_mw"] >= x["p10_mw"])
        & (x["target_power_mw"] <= x["p90_mw"])
    )

    x["absolute_error_mw"] = (
        x["target_power_mw"]
        - x["p50_mw"]
    ).abs()

    return x, robust_sigma


def evaluate(x, robust_sigma):

    mae = (
        x["absolute_error_mw"]
        .mean()
    )

    rmse = np.sqrt(
        (
            x["target_power_mw"]
            - x["p50_mw"]
        )
        .pow(2)
        .mean()
    )

    bias = (
        x["p50_mw"]
        - x["target_power_mw"]
    ).mean()

    coverage = (
        x["interval_covered"]
        .mean()
    )

    mean_width = (
        x["interval_width_mw"]
        .mean()
    )

    return {
        "point_mae_mw": round(
            float(mae), 3
        ),
        "point_rmse_mw": round(
            float(rmse), 3
        ),
        "bias_mw": round(
            float(bias), 3
        ),
        "target_coverage": TARGET_COVERAGE,
        "actual_coverage": round(
            float(coverage), 4
        ),
        "coverage_gap": round(
            float(coverage - TARGET_COVERAGE),
            4,
        ),
        "mean_interval_width_mw": round(
            float(mean_width), 3
        ),
        "robust_sigma_mw": round(
            float(robust_sigma), 3
        ),
    }


def print_report(x, metrics):

    print()
    print("=" * 60)
    print("FLOWER ENERGY PLATFORM")
    print("V9 PROBABILISTIC WIND FORECAST")
    print("=" * 60)

    print()
    print("============================================================")
    print("V9 RESULTS")
    print("============================================================")

    print(
        f"Point Forecast MAE : "
        f"{metrics['point_mae_mw']:.2f} MW"
    )

    print(
        f"Point Forecast RMSE: "
        f"{metrics['point_rmse_mw']:.2f} MW"
    )

    print(
        f"Bias               : "
        f"{metrics['bias_mw']:.2f} MW"
    )

    print(
        f"Target Coverage    : "
        f"{metrics['target_coverage']:.2%}"
    )

    print(
        f"Actual Coverage    : "
        f"{metrics['actual_coverage']:.2%}"
    )

    print(
        f"Coverage Gap       : "
        f"{metrics['coverage_gap']:.2%}"
    )

    print(
        f"Mean Interval Width: "
        f"{metrics['mean_interval_width_mw']:.2f} MW"
    )

    print(
        f"Robust Sigma       : "
        f"{metrics['robust_sigma_mw']:.2f} MW"
    )

    print()
    print("============================================================")
    print("RISK DISTRIBUTION")
    print("============================================================")

    print(
        x["risk_class"]
        .value_counts()
        .to_string()
    )

    print()
    print("============================================================")
    print("DISPATCH RECOMMENDATIONS")
    print("============================================================")

    print(
        x["dispatch_recommendation"]
        .value_counts()
        .to_string()
    )

    print()
    print("============================================================")
    print("WORST FORECASTS")
    print("============================================================")

    worst = (
        x.sort_values(
            "absolute_error_mw",
            ascending=False,
        )
        .head(10)
    )

    for _, row in worst.iterrows():

        print(
            f"{row['timestamp']} | "
            f"Actual={row['target_power_mw']:.2f} MW | "
            f"P50={row['p50_mw']:.2f} MW | "
            f"P10={row['p10_mw']:.2f} MW | "
            f"P90={row['p90_mw']:.2f} MW | "
            f"Risk={row['risk_class']}"
        )


def train():

    df = load_v8_predictions()

    if len(df) == 0:
        raise ValueError(
            "V8 prediction dataset is empty."
        )

    x, robust_sigma = (
        build_probabilistic_forecast(df)
    )

    metrics = evaluate(
        x,
        robust_sigma,
    )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    output_columns = [
        "timestamp",
        "asset_id",
        "wind_speed_forecast_ms",
        "target_power_mw",
        "p10_mw",
        "p50_mw",
        "p90_mw",
        "interval_width_mw",
        "downside_risk_mw",
        "upside_potential_mw",
        "prob_low_generation",
        "forecast_confidence",
        "risk_class",
        "dispatch_recommendation",
        "interval_covered",
        "absolute_error_mw",
    ]

    output = x[
        output_columns
    ].copy()

    prediction_path = (
        GOLD
        / "wind_probabilistic_predictions.parquet"
    )

    output.to_parquet(
        prediction_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = {
        "version": "V9",
        "model": "V8 calibrated probabilistic layer",
        "asset_id": ASSET_ID,
        "target_coverage": TARGET_COVERAGE,
        "capacity_mw": 1100.0,
        "quantiles": [
            "P10",
            "P50",
            "P90",
        ],
        "metrics": metrics,
        "rows": len(output),
        "prediction_path": str(
            prediction_path
        ),
    }

    metadata_path = (
        MODELS
        / "wind_probabilistic_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print_report(
        x,
        metrics,
    )

    print()
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