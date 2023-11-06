import requests
import json

data = [
    {
        'age': 39,
        'workclass': 'State-gov',
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

r = requests.post("https://income-classifier-wb9c.onrender.com/predict/", data=json.dumps(data))
print("status_code: ", r.status_code)
print("prediction: ", r.json())
