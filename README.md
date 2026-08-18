\# 🌱 Flower Energy Data Platform



\## End-to-End Data Engineering + AI/ML Platform for Renewable Energy Forecasting



Flower Energy Data Platform is an end-to-end renewable energy data platform built to demonstrate how a Data Engineer can design a reliable data pipeline that turns raw energy telemetry, weather forecasts, market data, and grid signals into ML-ready datasets, forecasts, model evaluation outputs, uncertainty estimates, and operational recommendations.



The project covers the full lifecycle:



\*\*Raw data → Bronze → Silver → Feature Engineering → ML Forecasting → Evaluation → Uncertainty → Risk → Gold Data Products\*\*



The main engineering focus is the data platform. Machine learning is treated as a workload that consumes the curated data rather than as an isolated notebook.



\---



\## 🏗️ Architecture



```text

                        ENERGY DATA SOURCES

                                 │

            ┌────────────────────┼────────────────────┐

            │                    │                    │

       Asset Telemetry      Weather Forecasts    Market / Grid

            │                    │                    │

            └────────────────────┼────────────────────┘

                                 ▼

                        DATA INGESTION LAYER

                                 │

                                 ▼

                             RAW DATA

                                 │

                                 ▼

                            BRONZE LAYER

                   Raw / standardized telemetry

                                 │

                                 ▼

                            SILVER LAYER

                Cleaned + validated analytical data

                                 │

                                 ▼

                      FEATURE ENGINEERING

         ┌───────────────────────┼───────────────────────┐

         │                       │                       │

    Historical              Weather                 Market

      Lags                  Features                Features

         │                       │                       │

         └───────────────────────┼───────────────────────┘

                                 ▼

                        ML TRAINING DATASET

                                 │

                ┌────────────────┴────────────────┐

                │                                 │

                ▼                                 ▼

         XGBoost Forecasting              Physics Model

                │                                 │

                └────────────────┬────────────────┘

                                 ▼

                        Hybrid / Residual ML

                                 │

                                 ▼

                        MODEL EVALUATION

                                 │

                 ┌───────────────┼────────────────┐

                 │               │                │

                MAE             RMSE             Bias

                 │               │                │

                 └───────────────┼────────────────┘

                                 ▼
                   UNCERTAINTY / CALIBRATION

                                 │

                          P10 / P50 / P90

                                 │

                                 ▼

                        RISK CLASSIFICATION

                                 │

                                 ▼
          
                             GOLD LAYER

                                 │

           ┌─────────────────────┼─────────────────────┐

           │                     │                     │

      Forecast Data        Evaluation Data       Risk Outputs

           │                     │                     │

           └─────────────────────┼─────────────────────┘

                                 ▼

                      BI / OPERATIONS / APIs

```



\---



\# 🎯 What I Built



The platform simulates three renewable-energy assets:



\- `SOLAR\_SE\_01`

\- `WIND\_SE\_01`

\- `BATTERY\_SE\_01`



The telemetry contains:



\- Timestamp

\- Asset ID

\- Power generation

\- Temperature

\- Wind speed

\- Cloud cover

\- Electricity market price

\- Grid imbalance

\- Battery state of charge



Weather forecasts are generated separately and joined to historical telemetry to create day-ahead forecasting datasets.



\---



\# 🔄 Data Engineering Pipeline



The platform follows a layered data architecture.



```text

                    ┌───────────────┐

                    │   RAW INPUT   │

                    └───────┬───────┘

                            │

                            ▼

                    ┌───────────────┐

                    │    BRONZE     │

                    │ Raw telemetry │

                    └───────┬───────┘

                            │

                       validation

                            │

                            ▼

                    ┌───────────────┐

                    │    SILVER     │

                    │ Cleaned data  │

                    └───────┬───────┘

                            │

                   transformations

                            │

                            ▼

                    ┌───────────────┐

                    │  FEATURES     │

                    │ ML-ready data │

                    └───────┬───────┘

                            │

                            ▼

                    ┌───────────────┐

                    │     GOLD      │

                    │ Data products │

                    └───────────────┘

```



The important design principle is that each stage has a clear responsibility.



\### Raw

Original generated telemetry is stored without applying business logic.



\### Bronze

The raw data is persisted as an intermediate ingestion layer.



\### Silver

The data is cleaned, standardized, ordered by asset and timestamp, and prepared for downstream transformations.



\### Gold

The Gold layer contains datasets intended for analytics, forecasting, monitoring, and operational consumption.



\---



\# 📊 Data Modeling



The telemetry model is centered around asset-level time-series observations.



```text

                 ASSET

                   │

                   │ asset\_id

                   ▼

              TELEMETRY

                   │

        ┌──────────┼──────────┐

        │          │          │

      Power     Weather     Market

        │          │          │

        └──────────┼──────────┘

                   │

                   ▼

             FEATURE DATASET

                   │

                   ▼

             MODEL OUTPUT

```



Core telemetry fields:



```text

timestamp

asset\_id

temperature\_c

wind\_speed\_ms

cloud\_cover

market\_price\_eur\_mwh

grid\_imbalance

power\_mw

state\_of\_charge

```



\---



\# ⚙️ Feature Engineering



The feature pipeline creates time-series and domain features at the asset level.



\### Historical generation



```text

power\_mw\_lag\_1

power\_mw\_lag\_24

power\_roll\_6

```



\### Market features



```text

market\_price\_eur\_mwh

market\_price\_eur\_mwh\_lag\_1

price\_roll\_6

grid\_imbalance

```



\### Calendar features



```text

hour

dayofweek

```



\### Weather features



```text

temperature\_forecast\_c

wind\_speed\_forecast\_ms

cloud\_cover\_forecast

```



\### Physics-informed features



```text

wind\_power\_curve\_factor

physics\_forecast\_mw

wind\_speed\_squared

wind\_speed\_cubed

historical\_capacity\_factor

historical\_capacity\_factor\_24h

```



\---



\# 🤖 Machine Learning



The forecasting layer uses XGBoost regression models.



The initial combined forecasting model was useful as a baseline, but the project evolved toward asset-specific models because solar, wind, and battery generation have very different behavior.



```text

                     ENERGY DATA

                          │

             ┌────────────┼────────────┐

             ▼            ▼            ▼

           SOLAR         WIND       BATTERY

             │            │            │

             ▼            ▼            ▼

          XGBoost       XGBoost      XGBoost

             │            │            │

             ▼            ▼            ▼

        Forecast       Forecast     Forecast

```



\---



\# ☀️ Solar Forecasting



The solar model uses historical generation, weather, calendar, and grid features.



Important features included:



```text

power\_mw\_lag\_24

power\_mw\_lag\_1

hour

cloud\_cover\_forecast

grid\_imbalance

power\_roll\_6

market\_price\_eur\_mwh

temperature\_forecast\_c

dayofweek

```



Observed result:



```text

MAE  : 28.16 MW

RMSE : 46.89 MW

```



The 24-hour lag became the strongest feature, which is consistent with the daily pattern of solar generation.



\---



\# 🌬️ Wind Forecasting



Wind forecasting was the most challenging part of the platform.



The model uses:



```text

wind\_speed\_forecast\_ms

power\_mw\_lag\_24

power\_mw\_lag\_1

hour

grid\_imbalance

power\_roll\_6

market\_price\_eur\_mwh

temperature\_forecast\_c

dayofweek

```



Initial asset-specific result:



```text

MAE  : 111.74 MW

RMSE : 168.82 MW

```



Wind speed was the dominant feature, matching the physical relationship between wind speed and turbine output.



Rather than hiding the weaker performance, I used it to investigate the model's failure modes.



\---



\# 🔬 Physics-Informed Wind Forecasting



A simplified turbine power curve was introduced to provide domain knowledge.



```text

Wind Speed

    │

    ▼

Cut-in Speed

    │

    ▼

Cubic Power Region

    │

    ▼

Rated Power

    │

    ▼

Cut-out Speed

    │

    ▼

Physics Forecast

```



Hybrid architecture:



```text

Weather Forecast

      │

      ├──────────────► Physics Model

      │                      │

      │                      ▼

      │               Physics Forecast

      │                      │

      └──────────────┐       │

                     ▼       ▼

                  ML Model

                     │

                     ▼

               Final Forecast

```



\---



\# 🧠 Residual Calibration



The next experiment used residual learning:



```text

Final Forecast = Physics Forecast + ML Correction

```



The ML model learns how the real system differs from the simplified physics model.



Observed experiment:



```text

Physics MAE : 356.66 MW

V6 MAE      : 212.90 MW



MAE improvement : 40.31%

```



\---



\# 📈 Model Evaluation \& Monitoring



The platform contains a dedicated evaluation layer.



Metrics include:



```text

MAE

RMSE

Bias

MAPE

Asset-level performance

Worst forecasts

Model health

```



Example:



```text

SOLAR\_SE\_01

MAE  = 28.16 MW

RMSE = 46.89 MW



WIND\_SE\_01

MAE  = 111.74 MW

RMSE = 168.82 MW



BATTERY\_SE\_01

MAE  = 5.27 MW

RMSE = 6.54 MW

```



The evaluation pipeline also surfaces the worst individual forecasts so that failures can be investigated rather than hidden inside aggregate metrics.



\---



\# 📊 Uncertainty-Aware Forecasting



Point forecasts do not tell an operator how uncertain the prediction is.



The platform therefore adds uncertainty estimation.



```text

P10 = 200 MW

P50 = 500 MW

P90 = 850 MW

```



The uncertainty layer evaluates:



```text

Target coverage

Actual coverage

Coverage gap

Mean interval width

Robust sigma

Forecast bias

```



Observed V9 result:



```text

Point Forecast MAE : 231.09 MW

Point Forecast RMSE: 298.65 MW

Actual Coverage    : 70.59%

Target Coverage    : 80.00%

Coverage Gap       : -9.41%

Mean Interval Width: 538.13 MW

```



This is treated as a monitoring signal rather than claiming that the uncertainty model is production-ready.



\---



\# ⚠️ Risk \& Dispatch Layer



Forecast uncertainty is converted into operational risk categories.



```text

Forecast

   │

   ▼

Uncertainty

   │

   ▼

Risk Classification

   │

   ├── NORMAL

   │       ↓

   │   NORMAL\_DISPATCH

   │

   ├── MEDIUM\_CONFIDENCE

   │       ↓

   │   MONITOR

   │

   ├── HIGH\_LOW\_GENERATION\_RISK

   │       ↓

   │   PREPARE\_RESERVE

   │

   └── CRITICAL\_LOW\_GENERATION

           ↓

       ACTIVATE\_RESERVE

```



This turns an ML output into a structured downstream data product that could be consumed by a dashboard, API, or operational workflow.



\---



\# 🗂️ Project Structure



```text

flower\_energy\_data\_platform/

│

├── data/

│   ├── raw/

│   │   └── energy\_observations.csv

│   ├── bronze/

│   ├── silver/

│   │   └── energy\_observations.parquet

│   └── gold/

│       ├── forecast\_predictions.parquet

│       ├── forecast\_evaluation.parquet

│       ├── wind\_physics\_predictions.parquet

│       ├── wind\_residual\_predictions.parquet

│       ├── wind\_uncertainty\_predictions.parquet

│       ├── wind\_calibrated\_predictions.parquet

│       └── wind\_probabilistic\_predictions.parquet

│

├── models/

│   ├── generation\_forecast.joblib

│   ├── wind\_physics\_day\_ahead.joblib

│   ├── wind\_residual\_calibration.joblib

│   ├── evaluation\_report.json

│   ├── wind\_physics\_metadata.json

│   ├── wind\_residual\_metadata.json

│   ├── wind\_uncertainty\_metadata.json

│   ├── wind\_calibration\_metadata.json

│   └── wind\_probabilistic\_metadata.json

│

├── src/

│   └── flower\_pipeline/

│       ├── config.py

│       ├── generate\_data.py

│       ├── features.py

│       ├── weather\_forecast.py

│       ├── day\_ahead.py

│       ├── asset\_day\_ahead.py

│       ├── asset\_models.py

│       ├── evaluation.py

│       ├── wind\_residual.py

│       ├── uncertainty.py

│       └── calibration.py

│

├── requirements.txt

├── pyproject.toml

└── README.md

```



\---



\# 🛠️ Technology Stack



\### Data Engineering

\- Python

\- Pandas

\- NumPy

\- Apache Parquet

\- Bronze / Silver / Gold architecture

\- Data validation

\- Data transformation

\- Time-series processing

\- Feature pipelines

\- Modular Python package structure



\### Machine Learning

\- XGBoost

\- Scikit-learn

\- Joblib

\- Regression

\- Feature importance

\- Residual learning

\- Physics-informed ML

\- Forecast evaluation

\- Uncertainty estimation



\### Engineering

\- Git

\- GitHub

\- Reproducible pipelines

\- JSON model metadata

\- Model artifacts



\---



\# ▶️ Running the Project



Create a virtual environment:



```bash

python -m venv .venv

```



Windows:



```powershell

.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Generate telemetry:



```bash

python -m flower\_pipeline.generate\_data

```



Generate weather forecasts:



```bash

python -m flower\_pipeline.weather\_forecast

```



Run day-ahead forecasting:



```bash

python -m flower\_pipeline.day\_ahead

```



Run asset-specific models:



```bash

python -m flower\_pipeline.asset\_day\_ahead

```



Run evaluation:



```bash

python -m flower\_pipeline.evaluation

```



Run wind residual calibration:



```bash

python -m flower\_pipeline.wind\_residual

```



Run uncertainty forecasting:



```bash

python -m flower\_pipeline.uncertainty

```



Run reliability calibration:



```bash

python -m flower\_pipeline.calibration

```



\---



\# 📦 Gold Data Products



```text

forecast\_predictions.parquet

forecast\_evaluation.parquet

wind\_physics\_predictions.parquet

wind\_residual\_predictions.parquet

wind\_uncertainty\_predictions.parquet

wind\_calibrated\_predictions.parquet

wind\_probabilistic\_predictions.parquet

```



Model metadata is stored separately in JSON files so configuration and evaluation results can be tracked alongside prediction datasets.



\---



\# 🎯 Why This Is a Data Engineering Project



The ML model is only one component.



The primary workflow is:



```text

Source Data

    ↓

Ingestion

    ↓

Storage

    ↓

Data Validation

    ↓

Transformation

    ↓

Feature Engineering

    ↓

Curated Data

    ↓

ML Training

    ↓

Predictions

    ↓

Evaluation

    ↓

Monitoring

    ↓

Operational Data Products

```



This demonstrates:



\- Building reliable data pipelines

\- Structuring data layers

\- Handling time-series data

\- Creating reusable transformations

\- Producing ML-ready datasets

\- Managing analytical data products

\- Tracking model outputs and metadata

\- Supporting downstream analytics and operational systems



The AI/ML layer demonstrates understanding of the workloads consuming the data platform.



\---



\# ☁️ Production Cloud Evolution



The current implementation is intentionally local and reproducible. The same logical architecture can be mapped to a cloud data platform:



```text

                  DATA SOURCES

                       │

                       ▼

                Azure Data Factory

                       │

                       ▼

                 ADLS Gen2

                       │

                       ▼

                 BRONZE / DELTA

                       │

                       ▼

             Microsoft Fabric / Spark

                       │

                       ▼

                 SILVER / DELTA

                       │

                       ▼

             Feature Engineering

                       │

             ┌─────────┴─────────┐

             ▼                   ▼

        ML Features         BI Features

             │                   │

             ▼                   ▼

         Azure ML             Power BI

             │

             ▼

      Forecasting Models

             │

             ▼

      Model Evaluation

             │

             ▼

      Uncertainty / Risk

             │

             ▼

            GOLD

```



Potential production components:



\- Azure Data Factory

\- ADLS Gen2

\- Microsoft Fabric

\- OneLake

\- Delta Lake

\- PySpark

\- Azure Machine Learning

\- MLflow

\- Power BI

\- GitHub Actions

\- Docker

\- FastAPI



\---



\# 🔮 Future Improvements



\- Replace simulated telemetry with real energy APIs

\- Add Azure Data Factory ingestion

\- Move storage to ADLS Gen2 / OneLake

\- Convert processing to PySpark

\- Use Delta Lake for transactional storage

\- Add incremental processing

\- Add automated data-quality checks

\- Add MLflow experiment tracking

\- Add model and feature drift monitoring

\- Add automated retraining

\- Add CI/CD with GitHub Actions

\- Containerize the pipeline

\- Deploy forecasting APIs with FastAPI

\- Build a Power BI operational dashboard

\- Integrate real weather and electricity-market data



\---



\# 📚 Key Lessons



The most important lesson from the project was that improving the model is only part of the problem.



The wind forecasting experiments showed that:



\- Data quality directly affects model quality.

\- Weather forecast quality is critical.

\- Physical constraints can improve model design.

\- Different assets require different modeling strategies.

\- Aggregate metrics can hide serious individual forecast failures.

\- Prediction intervals need calibration.

\- Monitoring is necessary after model training.

\- Good ML depends on good data engineering.



The project therefore evolved from a simple forecasting model into a broader platform:



```text

Reliable Data

      ↓

Good Data Engineering

      ↓

Meaningful Features

      ↓

Responsible ML

      ↓

Forecast Monitoring

      ↓

Risk-Aware Outputs

      ↓

Operational Decisions

```



\---



\# 👨‍💻 About



Built by \*\*Mokshagna Guntamadugu\*\* as a hands-on project combining Data Engineering, Analytics, Machine Learning, and Renewable Energy.



The goal is to demonstrate how a modern data platform can support AI/ML workloads from ingestion all the way to operational decision-making.



\*\*Data Engineering first. AI/ML where it adds value.\*\*



