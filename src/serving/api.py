from fastapi import FastAPI
import pandas
import cloudpickle as pickle
from src.serving.data_models import Settings, InputData, Prediction
from src.config import TARGET_MAPS

app = FastAPI()


from src.config import FEATURES

settings = Settings()

with open(settings.MODEL_PATH, "rb") as pickle_model:
    pipeline_model = pickle.load(pickle_model)


@app.get("/health")
async def health() -> int:
    return 200


@app.post("/predict")
async def predict(input_data: InputData) -> Prediction:
    df_input = pandas.DataFrame.from_dict(input_data.model_dump(), orient="index").T
    predictions = pipeline_model.predict(df_input[FEATURES])

    return {
        "id": input_data.id,
        "price_category": [TARGET_MAPS[pred] for pred in predictions][0],
    }
