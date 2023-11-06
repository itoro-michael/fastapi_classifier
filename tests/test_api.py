import pytest

from starter.ml.data import dict_greeting, list_of_dict_request
from fastapi.testclient import TestClient
from main import app

# Instantiate the testing client with app.
client = TestClient(app)


@pytest.fixture
def list_of_dict_request_2():
    list_of_dict_request_2 = [
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
            'capital_gain': 20174,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country': ' United-States'
        }
    ]
    return list_of_dict_request_2


# Test root "/" endpoint
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == dict_greeting


# Test POST method with "/items" endpoint
def test_single_prediction():
    r = client.post(
        "/predict/",
        json=list_of_dict_request,
    )
    assert r.status_code == 200
    assert r.json() == {
        "prediction": ["<=50K"]
    }


def test_multiple_predictions(list_of_dict_request_2):
    r = client.post(
        "/predict/",
        json=list_of_dict_request + list_of_dict_request_2,
    )
    assert r.status_code == 200
    assert r.json() == {
        "prediction": ["<=50K", ">50K"]
    }
