import pandas as pd


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Return 'classification' or 'regression' based on the target column."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    target = df[target_col]

    if pd.api.types.is_numeric_dtype(target):
        # Heuristic: low unique count relative to rows → treat as classification
        # (e.g. a 0/1 label stored as int, or a 1-5 rating scale)
        n_unique = target.nunique()
        if n_unique <= 10 and n_unique / len(target) < 0.05:
            return "classification"
        return "regression"
    else:
        return "classification"