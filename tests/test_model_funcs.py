import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier
from starter.ml import model, data


@pytest.fixture
def classifier():
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42)
    return clf


def test_inference(classifier, data_df, categorical_features):
    """Tests inference function"""
    X_train, y_train, encoder, lb = data.process_data(
        data_df, categorical_features=categorical_features, label='salary', training=True)
    classifier.fit(X_train[:100], y_train[:100])  # Test training
    pred = model.inference(classifier, X_train)
    assert isinstance(pred, np.ndarray), "pred should be np.ndarray."


def test_save_model(classifier, tmp_path):
    """Test the save_model method"""
    dir_path = tmp_path / "test"
    dir_path.mkdir()
    file_path = dir_path / "model.pkl"
    model.save_model(classifier, file_path)
    assert file_path.is_file(), "Review model.save_model()."


def test_load_model(classifier, tmp_path):
    """Test the load_model method"""
    dir_path = tmp_path / "test"
    dir_path.mkdir()
    file_path = dir_path / "model.pkl"
    model.save_model(classifier, file_path)
    clf = model.load_model(file_path)
    assert isinstance(clf, RandomForestClassifier), "Review model.load_model()."
