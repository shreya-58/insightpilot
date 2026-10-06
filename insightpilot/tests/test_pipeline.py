import pandas as pd
from pipeline.profiler import profile_dataframe

def test_profile_basic():
    df = pd.DataFrame({
        "age": [25, 30, None, 40],
        "category": ["a", "b", "a", "c"],
    })
    profile = profile_dataframe(df)
    assert profile["n_rows"] == 4
    assert profile["columns"]["age"]["null_count"] == 1
    assert profile["columns"]["category"]["cardinality"] == 3