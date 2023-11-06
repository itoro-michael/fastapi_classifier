import pandas as pd
import numpy as np
import pytest

from starter.ml.data import process_data, trim_dataframe
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.absolute() / 'data'


@pytest.fixture
def data():
    df = pd.read_csv(f'{DATA_DIR}/census.csv')
    df = trim_dataframe(df)
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


def test_data_types(data, categorical_features):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=categorical_features, label="salary", training=True)
    assert isinstance(X_train, np.ndarray), "np.ndarray required for X_train."
    assert isinstance(X_train, np.ndarray), "np.ndarray required for y_train."


def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000, "Dataset size is outside limits."


def test_cat_features(data):
    expected_colums = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert all([col in these_columns for col in expected_colums]
               ), "Missing columnss."
