from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    """Prediction response model"""
    base_name:str
    class_index:int
    class_name:str
    confidence:float


class PredictionsResponses(BaseModel):
    """Predictions responses model"""
    predictions: List[PredictionResponse]