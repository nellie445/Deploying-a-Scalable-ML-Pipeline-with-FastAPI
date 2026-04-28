import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, train_model


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_process_data():
    """
    Test that process_data returns non-empty training data and matching labels.
    """
    data = pd.read_csv("data/census.csv")
    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0
    assert encoder is not None
    assert lb is not None


def test_train_model():
    """
    Test that train_model returns a RandomForestClassifier.
    """
    data = pd.read_csv("data/census.csv")
    train = data.sample(n=1000, random_state=42)

    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """
    Test that precision, recall, and fbeta are floats within valid range.
    """
    y = [1, 0, 1, 1]
    preds = [1, 0, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1