import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GOLD = ROOT / "data" / "gold"
MODELS = ROOT / "models"

INPUT = GOLD / "wind_residual_predictions.parquet"
OUTPUT = GOLD / "wind_calibrated_predictions.parquet"
METADATA = MODELS / "wind_calibration_metadata.json"

TARGET_COVERAGE = 0.80


def conformal_quantile(values, coverage=0.80):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    n = len(values)

    rank = int(
        np.ceil((n + 1) * coverage)
    )

    rank = min(max(rank, 1), n)

    sorted_values = np.sort(values)

    return float(sorted_values[rank - 1])


def load_data():

    df = pd.read_parquet(INPUT)

    print("Input columns:")
    print(df.columns.tolist())

    required = [
        "timestamp",
        "asset_id",
        "wind_speed_forecast_ms",
        "target_power_mw",
        "final_prediction_mw",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def calibrate():

    df = load_data()

    print()
    print("=" * 60)
    print("FLOWER ENERGY PLATFORM")
    print("V8 FORECAST RELIABILITY CALIBRATION")
    print("=" * 60)

    print(f"V7 rows: {len(df)}")

    # ---------------------------------------------------------
    # Calculate V7 absolute residual
    # ---------------------------------------------------------

    df["absolute_residual_mw"] = (
        df["target_power_mw"]
        - df["final_prediction_mw"]
    ).abs()

    # ---------------------------------------------------------
    # Wind regimes
    # ---------------------------------------------------------

    df["wind_regime"] = pd.cut(
        df["wind_speed_forecast_ms"],
        bins=[
            -np.inf,
            6,
            10,
            14,
            np.inf,
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
            "EXTREME",
        ],
    )

    # ---------------------------------------------------------
    # Chronological calibration/test split
    # ---------------------------------------------------------

    split = int(len(df) * 0.80)

    calibration_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    print(
        f"Calibration rows: {len(calibration_df)}"
    )

    print(
        f"Testing rows    : {len(test_df)}"
    )

    # ---------------------------------------------------------
    # Global conformal radius
    # ---------------------------------------------------------

    global_radius = conformal_quantile(
        calibration_df["absolute_residual_mw"],
        TARGET_COVERAGE,
    )

    # ---------------------------------------------------------
    # Regime-specific radii
    # ---------------------------------------------------------

    regime_radii = {}

    for regime in [
        "LOW",
        "MEDIUM",
        "HIGH",
        "EXTREME",
    ]:

        values = calibration_df.loc[
            calibration_df["wind_regime"] == regime,
            "absolute_residual_mw",
        ]

        if len(values) >= 10:

            radius = conformal_quantile(
                values,
                TARGET_COVERAGE,
            )

        else:

            radius = global_radius

        regime_radii[regime] = radius

    # ---------------------------------------------------------
    # Apply calibration
    # ---------------------------------------------------------

    test_df["conformal_radius_mw"] = (
        test_df["wind_regime"]
        .map(regime_radii)
        .astype(float)
    )

    test_df["lower_bound_mw"] = (
        test_df["final_prediction_mw"]
        - test_df["conformal_radius_mw"]
    )

    test_df["upper_bound_mw"] = (
        test_df["final_prediction_mw"]
        + test_df["conformal_radius_mw"]
    )

    # Physical wind farm limits
    test_df["lower_bound_mw"] = (
        test_df["lower_bound_mw"]
        .clip(lower=0)
    )

    test_df["upper_bound_mw"] = (
        test_df["upper_bound_mw"]
        .clip(upper=1100)
    )

    # ---------------------------------------------------------
    # Coverage
    # ---------------------------------------------------------

    test_df["covered"] = (
        (test_df["target_power_mw"]
         >= test_df["lower_bound_mw"])
        &
        (test_df["target_power_mw"]
         <= test_df["upper_bound_mw"])
    )

    test_df["interval_width_mw"] = (
        test_df["upper_bound_mw"]
        - test_df["lower_bound_mw"]
    )

    # ---------------------------------------------------------
    # Point forecast metrics
    # ---------------------------------------------------------

    error = (
        test_df["target_power_mw"]
        - test_df["final_prediction_mw"]
    )

    mae = error.abs().mean()

    rmse = np.sqrt(
        (error ** 2).mean()
    )

    bias = error.mean()

    coverage = (
        test_df["covered"].mean()
        * 100
    )

    mean_width = (
        test_df["interval_width_mw"]
        .mean()
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("V8 RESULTS")
    print("=" * 60)

    print(
        f"Point Forecast MAE : {mae:.2f} MW"
    )

    print(
        f"Point Forecast RMSE: {rmse:.2f} MW"
    )

    print(
        f"Bias               : {bias:.2f} MW"
    )

    print(
        f"Target Coverage    : "
        f"{TARGET_COVERAGE * 100:.2f}%"
    )

    print(
        f"Actual Coverage    : "
        f"{coverage:.2f}%"
    )

    print(
        f"Mean Interval Width: "
        f"{mean_width:.2f} MW"
    )

    # ---------------------------------------------------------
    # Regime performance
    # ---------------------------------------------------------

    print()
    print("REGIME CALIBRATION")
    print("-" * 60)

    for regime, radius in regime_radii.items():

        subset = test_df[
            test_df["wind_regime"] == regime
        ]

        if len(subset) == 0:
            continue

        regime_coverage = (
            subset["covered"].mean()
            * 100
        )

        print(
            f"{regime:8s} | "
            f"rows={len(subset):3d} | "
            f"radius={radius:.2f} MW | "
            f"coverage={regime_coverage:.2f}%"
        )

    # ---------------------------------------------------------
    # Worst forecasts
    # ---------------------------------------------------------

    test_df["absolute_error_mw"] = error.abs()

    worst = (
        test_df
        .sort_values(
            "absolute_error_mw",
            ascending=False,
        )
        .head(10)
    )

    print()
    print("=" * 60)
    print("WORST FORECASTS")
    print("=" * 60)

    for _, row in worst.iterrows():

        print(
            f"{row['timestamp']} | "
            f"Actual={row['target_power_mw']:.2f} MW | "
            f"Forecast={row['final_prediction_mw']:.2f} MW | "
            f"Interval=["
            f"{row['lower_bound_mw']:.2f}, "
            f"{row['upper_bound_mw']:.2f}] | "
            f"Covered={row['covered']}"
        )

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    test_df.to_parquet(
        OUTPUT,
        index=False,
    )

    metadata = {
        "version": "V8",
        "method": "regime_based_conformal_calibration",
        "target_coverage": TARGET_COVERAGE,
        "actual_coverage": round(
            float(coverage),
            3,
        ),
        "mean_interval_width_mw": round(
            float(mean_width),
            3,
        ),
        "mae_mw": round(
            float(mae),
            3,
        ),
        "rmse_mw": round(
            float(rmse),
            3,
        ),
        "bias_mw": round(
            float(bias),
            3,
        ),
        "global_radius_mw": round(
            float(global_radius),
            3,
        ),
        "regime_radii_mw": {
            key: round(float(value), 3)
            for key, value in regime_radii.items()
        },
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print(
        f"Predictions saved: {OUTPUT}"
    )

    print(
        f"Metadata saved: {METADATA}"
    )


if __name__ == "__main__":
    calibrate()