import cloudpickle
import numpy as np

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train:np.ndarray, y_train:np.ndarray) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    return clf    


def compute_model_metrics(y:np.ndarray, preds:np.ndarray):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
 
 
def save_model(model, file_path:str) -> None:
    """
    Saves the machine learning model to file_path.

    Inputs
    ------
    model : Object
        Training data.
    file_path : str
        Model path (.pkl).
    Returns
    -------
    None
    """
    with open(file_path, mode='wb') as file:
        cloudpickle.dump(model, file)
        
        
def load_model(file_path:str) -> RandomForestClassifier:
    """
    Loads the machine learning model from file_path.

    Inputs
    ------
    file_path : str
        Model path (.pkl).
    Returns
    -------
    model : RandomForestClassifier
    """
    with open(file_path, mode='rb') as file:
        model = cloudpickle.load(file)
    return model