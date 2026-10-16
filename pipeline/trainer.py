import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
import time


def prepare_data(df: pd.DataFrame, target_col: str):
    """Basic cleanup: drop rows with missing target, encode categoricals, split."""
    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode any categorical feature columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna("missing").astype(str)
            X[col] = LabelEncoder().fit_transform(X[col])

    # Fill remaining numeric nulls with column median
    X = X.fillna(X.median(numeric_only=True))

    # Encode target if classification with string labels
    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, list(X.columns)



def get_candidate_models(task_type: str) -> dict:
    """Hardcoded baseline models for now — replaced by LLM Advisor later."""
    if task_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
        }


def train_and_evaluate(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, target_col)
    models = get_candidate_models(task_type)

    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start, 3)
        preds = model.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "f1": round(f1_score(y_test, preds, average="weighted"), 4),
            }
        else:
            metrics = {
                "rmse": round(mean_squared_error(y_test, preds) ** 0.5, 4),
                "r2": round(r2_score(y_test, preds), 4),
            }

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_names, model.feature_importances_.round(4)))
            importances = dict(
                sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            )

        results[name] = {
            "metrics": metrics,
            "train_time_sec": train_time,
            "top_features": importances,
        }

    return results