from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='APP_')
    MODEL_PATH: str = '/opt/ml/model/model.pkl'  # will be read from `APP_MODEL_PATH`

class InputData(BaseModel):
    id: int
    accommodates: int
    room_type: str
    beds: int
    bedrooms: int
    bathrooms: int
    neighbourhood: Literal["Bronx", "Queens", "Staten Island", "Brooklyn", "Manhattan"]
    tv: int
    elevator: int
    internet: int
    latitude: float
    longitude: float

class Prediction(BaseModel):
    id: int
    price_category: Literal["low", "mid", "high", "lux"]