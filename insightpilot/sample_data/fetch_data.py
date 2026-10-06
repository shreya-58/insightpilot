# scratch script to generate CSVs from sklearn
from sklearn.datasets import load_iris, fetch_california_housing
import pandas as pd

iris = load_iris(as_frame=True)
iris.frame.to_csv("sample_data/iris.csv", index=False)

housing = fetch_california_housing(as_frame=True)
housing.frame.to_csv("sample_data/california_housing.csv", index=False)