import pytest
import pandas as pd

from pathlib import Path
from starter.ml import data

DATA_DIR = Path(__file__).parent.parent.absolute() / 'data'


@pytest.fixture
def data_df():
    df = pd.read_csv(f'{DATA_DIR}/census.csv')
    df = data.trim_dataframe(df)
    return df


@pytest.fixture
def categorical_features():
    cat_colums = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    return cat_colums
