from datetime import datetime
from pydantic import BaseModel

class ForecastResponse(BaseModel):
    timestamp: datetime
    asset_id: str
    power_mw: float
    prediction_mw: float
    absolute_error_mw: float
