import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from pydantic import BaseModel
from typing import Union

list_metric = ['column_name', 'column_value', 'precision', 'recall', 'fbeta', 'accuracy']

dict_pred = {"0": "<=50K", "1": ">50K"}

list_of_dict_request = [
    {
        'age': 39,
        'workclass': ' State-gov',
                     'fnlgt': 77516,
                     'education': ' Bachelors',
                     'education_num': 13,
                     'marital_status': ' Never-married',
                     'occupation': ' Adm-clerical',
                     'relationship': ' Not-in-family',
                     'race': ' White',
                     'sex': ' Male',
                     'capital_gain': 2174,
                     'capital_loss': 0,
                     'hours_per_week': 40,
                     'native_country': ' United-States'
    }
]


class Data(BaseModel):
    # list_of_dict: list
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str
    salary: Union[str, None] = None

    model_config = {
        "json_schema_extra": {
            "examples": list_of_dict_request
        }
    }


list_required_columns = [
    'age',
    'workclass',
    'fnlgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country']


list_categorical_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


dict_greeting = {
    "greeting": "Get predictions using the /predict POST endpoint"}


def trim_dataframe(data_df: pd.DataFrame) -> pd.DataFrame:
    """Removes leading and trailing spaces from column names and data elements."""
    data_df.columns = [col.strip().replace('-', '_')
                       for col in data_df.columns]
    data_df = data_df.applymap(
        lambda x: x.strip() if isinstance(
            x, str) else x)
    return data_df


def dataset_with_selected_column_value(
        X: pd.DataFrame,
        label: str,
        column_value: str) -> pd.DataFrame:
    output = X[X[label] == column_value]
    return output


def process_data(
    X: pd.DataFrame,
    categorical_features: list = [],
    label: str = None,
    training: bool = True,
    encoder: OneHotEncoder = None,
    lb: LabelBinarizer = None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
