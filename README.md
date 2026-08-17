# Flower Energy Data & AI Platform

A production-style portfolio project inspired by Flower's renewable-energy data platform problem.

It ingests synthetic battery/solar/wind telemetry plus weather, grid and market signals; validates
and normalizes the data through Bronze/Silver/Gold layers; builds ML features; forecasts generation;
stores predictions; and serves them through FastAPI.

## Architecture

Sources -> Python ingestion -> Bronze -> Silver -> Gold/dbt -> ML features -> Forecast + anomaly-ready
analytics -> FastAPI -> CI/Docker/Terraform boundary

## Why this fits Flower

Flower's Data Engineer roles emphasize scalable pipelines, Databricks/dbt/Python, Spark, CI/CD,
IaC, observability, data quality and energy datasets such as live grid signals, weather forecasts,
market data and asset telemetry. This project demonstrates those concepts with safe synthetic data.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

set PYTHONPATH=src
python -m flower_pipeline.generate_data
python -m flower_pipeline.pipeline
python -m flower_pipeline.train
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Test

```bash
pytest -q
```

## Docker

```bash
docker build -t flower-energy-platform .
docker run -p 8000:8000 flower-energy-platform
```

## Interview explanation

"I built a production-style energy data platform around Flower's problem space. I simulated
telemetry, weather, grid and market feeds, applied data contracts and medallion transformations,
engineered time-aware forecasting features, trained a generation model without future leakage,
and exposed model results through an API. I added tests, CI, Docker and a Terraform boundary so the
local design can migrate to Databricks/Delta and cloud infrastructure."

## Production evolution

1. CSV -> Kafka/Event Hubs/API connectors.
2. Local files -> ADLS/S3 + Delta Lake.
3. Local transforms -> Databricks/Spark.
4. SQL models -> dbt on Databricks SQL.
5. Add Airflow/Dagster orchestration and SLA monitoring.
6. Add Great Expectations/Pandera and lineage.
7. Add MLflow model registry and feature store.
8. Add real-time inference.
9. Add battery dispatch optimization using forecasts, prices and constraints.
