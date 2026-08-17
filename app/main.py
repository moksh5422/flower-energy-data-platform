from pathlib import Path
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from flower_pipeline.config import MODELS,GOLD
from .schemas import ForecastResponse

app = FastAPI(title="Flower Energy Data Platform",version="1.0.0")

@app.get("/health")
def health():
    return {"status":"ok","service":"flower-energy-platform"}

@app.get("/model")
def model_info():
    p = MODELS/"metadata.json"
    if not p.exists():
        raise HTTPException(503,"Model metadata unavailable")
    return json.loads(p.read_text())

@app.get("/forecast",response_model=list[ForecastResponse])
def forecast(limit:int=25):
    p = GOLD/"forecast_predictions.parquet"
    if not p.exists():
        raise HTTPException(503,"Forecast unavailable. Run pipeline and training first.")
    return pd.read_parquet(p).tail(limit).to_dict(orient="records")
