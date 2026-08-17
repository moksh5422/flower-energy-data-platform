from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
MODELS = ROOT / "models"

for p in [RAW, BRONZE, SILVER, GOLD, MODELS]:
    p.mkdir(parents=True, exist_ok=True)
