import pandas as pd
import logging

from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import List
from starter.ml import model, data

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO)
MODEL_DIR = Path(__file__).parent.absolute() / 'model'

# Load model
classifier = model.load_model(f'{MODEL_DIR}/model.pkl')
encoder = model.load_model(f'{MODEL_DIR}/encoder.pkl')
lb = model.load_model(f'{MODEL_DIR}/lb.pkl')


app = FastAPI(
    title="Income Classifier API",
    description="An API that predicts income level.",
    version="1.0.0",
)


@app.post("/predict/")
async def predict_income(input: List[data.Data]):
    # Obtain inputs
    list_data = [item.model_dump() for item in input]  # model_dump or dict()

    # Handle empty list and None
    if not list_data:
        raise HTTPException(status_code=400,
                            detail="Empty list of dictionaries provided.")

    # Check for required columns
    list_col = [col.strip() for col in list_data[0].keys()]
    if not all([col in list_col for col in data.list_required_columns]):
        raise HTTPException(
            status_code=400,
            detail=f"Request missing required fields - {data.list_required_columns}")

    # Convert to dataframe
    request_df = pd.DataFrame(list_data)

    # Clean data
    request_df = data.trim_dataframe(request_df)

    # Drop salary column if given
    if 'salary' in request_df.columns:
        request_df = request_df.drop(['salary'], axis=1)

    # Get inference
    cat_features = data.list_categorical_features

    # Proces the request data with the process_data function.
    logging.info('Calling the process_data function, with training=False.')
    X, y, _, _ = data.process_data(request_df,
                                   categorical_features=cat_features,
                                   label=None,
                                   training=False,
                                   encoder=encoder,
                                   lb=lb)
    logging.info('process_data function call completed')

    # Print model metric
    logging.info('Get inference.')
    pred = model.inference(classifier, X)
    logging.info(f'Inference obtained. Prediction size = {len(pred)}')

    list_pred = [data.dict_pred[str(val)] for val in pred]
    return {"prediction": list_pred}


@app.get("/")
async def hello_world():
    return data.dict_greeting
