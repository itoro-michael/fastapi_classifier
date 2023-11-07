import numpy as np

from starter.ml.data import process_data
from sklearn.model_selection import train_test_split


def test_data_types(data_df, categorical_features):
    train, test = train_test_split(data_df, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=categorical_features, label="salary", training=True)
    assert isinstance(X_train, np.ndarray), "np.ndarray required for X_train."
    assert isinstance(X_train, np.ndarray), "np.ndarray required for y_train."


def test_row_count(data_df):
    assert 15000 < data_df.shape[0] < 1000000, "Dataset size is outside limits."


def test_cat_features(data_df):
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

    these_columns = data_df.columns.values

    # This also enforces the same order
    assert all([col in these_columns for col in expected_colums]
               ), "Missing columns."
