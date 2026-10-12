import pandas as pd
from pipeline.profiler import profile_dataframe
from pipeline.task_detector import detect_task_type
from pipeline.trainer import train_and_evaluate

def test_profile_basic():
    df = pd.DataFrame({
        "age": [25, 30, None, 40],
        "category": ["a", "b", "a", "c"],
    })
    profile = profile_dataframe(df)
    assert profile["n_rows"] == 4
    assert profile["columns"]["age"]["null_count"] == 1
    assert profile["columns"]["category"]["cardinality"] == 3
    

def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Return 'classification' or 'regression' based on the target column."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    target = df[target_col]

    if pd.api.types.is_numeric_dtype(target):
        n_unique = target.nunique()
        if n_unique <= 10:
            return "classification"
        if n_unique / len(target) < 0.05:
            return "classification"
        return "regression"
    else:
        return "classification"
    

def test_detect_task_type_classification():
    df = pd.DataFrame({"target": [0, 1, 0, 1, 1, 0, 1, 0] * 3, "x": range(24)})
    assert detect_task_type(df, "target") == "classification"


def test_detect_task_type_regression():
    df = pd.DataFrame({"target": [i * 1.5 for i in range(50)], "x": range(50)})
    assert detect_task_type(df, "target") == "regression"


def test_train_and_evaluate_classification_runs():
    df = pd.DataFrame({
        "feature1": list(range(100)),
        "feature2": [i % 3 for i in range(100)],
        "target": [i % 2 for i in range(100)],
    })
    results = train_and_evaluate(df, "target", "classification")
    assert "RandomForest" in results
    assert "accuracy" in results["RandomForest"]["metrics"]
    
    
# tests/test_pipeline.py (append to the existing file)

from unittest.mock import MagicMock
from pipeline.advisor import get_model_recommendation


def test_get_model_recommendation_parses_valid_json():
    mock_client = MagicMock()
    mock_client.call.return_value = '{"models": [{"name": "RandomForest", "reasoning": "robust"}], "preprocessing_notes": "none needed"}'

    result = get_model_recommendation("classification", 150, 5, "clean data", mock_client, model="test")
    assert result.models[0].name == "RandomForest"


def test_get_model_recommendation_falls_back_on_bad_json():
    mock_client = MagicMock()
    mock_client.call.return_value = "not json at all"

    result = get_model_recommendation("classification", 150, 5, "clean data", mock_client, model="test")
    assert len(result.models) >= 1