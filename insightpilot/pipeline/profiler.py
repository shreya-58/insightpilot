import pandas as pd
import numpy as np
import json
from pathlib import Path

PROMPT_PATH = Path("llm/prompts/profiler_prompt.txt")


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame, raising a clear error if it fails."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Could not read CSV at {filepath}: {e}")
    if df.empty:
        raise ValueError("Uploaded CSV has no rows.")
    return df


def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        col_info = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_info.update({
                "mean": float(series.mean()) if series.notna().any() else None,
                "std": float(series.std()) if series.notna().any() else None,
                "min": float(series.min()) if series.notna().any() else None,
                "max": float(series.max()) if series.notna().any() else None,
                "skew": float(series.skew()) if series.notna().any() else None,
            })
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            col_info["outlier_count"] = int(outliers.count())
        else:
            col_info.update({
                "cardinality": int(series.nunique()),
                "top_values": series.value_counts().head(5).to_dict(),
            })

        profile["columns"][col] = col_info

    return profile



def generate_profile_summary(profile: dict, client, model: str) -> str:
    """Call the LLM (via Prune) to summarize profiling stats in plain English."""
    prompt_template = PROMPT_PATH.read_text()
    user_prompt = prompt_template.format(profile_json=json.dumps(profile, default=str))

    summary = client.call(
        model=model,
        system_prompt="You are a careful, concise data quality analyst.",
        user_prompt=user_prompt,
        max_tokens=400,
        stage="profiler",
    )
    return summary