from pydantic import BaseModel

class WaterQualityInput(BaseModel):
    Sulfate: float
    ph: float
    Chloramines: float
    Solids: float