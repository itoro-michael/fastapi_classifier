# Income Prediction API

Predicts the income of a person when required data is provided. Uses the random
forest classifier from sklearn. Model has accuracy of 85% on test set.

## Sample input
List of dictionaries.

```
[
  {
    'age': 39,
    'workclass': ' State-gov',
    'fnlgt': 77516,
    'education': ' Bachelors',
    'education-num': 13,
    'marital-status': ' Never-married',
    'occupation': ' Adm-clerical',
    'relationship': ' Not-in-family',
    'race': ' White',
    'sex': ' Male',
    'capital-gain': 2174,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': ' United-States'
  }
 ]
```

## Sample output
`<=50k`

`>50k`

## Getting Started

To run project use the command:

`uvicorn main:app --host=localhost --port=5000`


To interact with the API, navigate to: 


http://localhost:5000/docs


### Model Training

To train the model run the command:

`python train.py`


