import pandas as pd
import logging

from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import List

from starter.ml.model import load_model, inference
from starter.ml.data import process_data, Data, list_required_columns, trim_dataframe, \
    list_categorical_features, dict_greeting, dict_pred

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO)
MODEL_DIR = Path(__file__).parent.absolute() / 'model'

# Load model
model = load_model(f'{MODEL_DIR}/model.pkl')
encoder = load_model(f'{MODEL_DIR}/encoder.pkl')
lb = load_model(f'{MODEL_DIR}/lb.pkl')


app = FastAPI(
    title="Income Classifier API",
    description="An API that predicts income level.",
    version="1.0.0",
)


@app.post("/predict/")
async def predict_income(data: List[Data]):
    # Obtain inputs
    list_data = [item.model_dump() for item in data]  # model_dump or dict()

    # Handle empty list and None
    if not list_data:
        raise HTTPException(status_code=400,
                            detail="Empty list of dictionaries provided.")

    # Check for required columns
    list_col = [col.strip() for col in list_data[0].keys()]
    if not all([col in list_col for col in list_required_columns]):
        raise HTTPException(
            status_code=400,
            detail=f"Request missing required fields - {list_required_columns}")

    # Convert to dataframe
    request_df = pd.DataFrame(list_data)

    # Clean data
    request_df = trim_dataframe(request_df)

    # Drop salary column if given
    if 'salary' in request_df.columns:
        request_df = request_df.drop(['salary'], axis=1)

    # Get inference
    cat_features = list_categorical_features

    # Proces the request data with the process_data function.
    logging.info('Calling the process_data function, with training=False.')
    X, y, _, _ = process_data(request_df,
                              categorical_features=cat_features,
                              label=None,
                              training=False,
                              encoder=encoder,
                              lb=lb)
    logging.info('process_data function call completed')

    # Print model metric
    logging.info('Get inference.')
    pred = inference(model, X)
    logging.info(f'Inference obtained. Prediction size = {len(pred)}')

    list_pred = [dict_pred[str(val)] for val in pred]
    return {"prediction": list_pred}


@app.get("/")
async def hello_world():
    return dict_greeting
