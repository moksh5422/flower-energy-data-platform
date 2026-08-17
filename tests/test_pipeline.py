import pandas as pd
from src.flower_pipeline.quality import validate
from src.flower_pipeline.features import make_features

def test_quality_contract():
    df = pd.DataFrame({
        "timestamp":pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "asset_id":["SOLAR_SE_01"],"temperature_c":[5],"wind_speed_ms":[6],
        "cloud_cover":[.2],"market_price_eur_mwh":[80],"grid_imbalance":[.1],
        "power_mw":[100],"state_of_charge":[None]
    })
    assert validate(df)["rows"] == 1

def test_feature_generation():
    rows=[]
    for i in range(30):
        rows.append({
            "timestamp":pd.Timestamp("2026-01-01",tz="UTC")+pd.Timedelta(hours=i),
            "asset_id":"WIND_SE_01","temperature_c":5,"wind_speed_ms":7,
            "cloud_cover":.2,"market_price_eur_mwh":80+i,
            "grid_imbalance":.1,"power_mw":100+i,"state_of_charge":None
        })
    out=make_features(pd.DataFrame(rows))
    assert len(out)>0
    assert "power_mw_lag_24" in out.columns
