
import pandas as pd
import json
from train import ORIGINAL_FEATURE_NAMES

df = pd.read_csv(
    "data/biodeg.csv",
    sep=";",
    header=None
)

sample = dict(zip(
    ORIGINAL_FEATURE_NAMES,
    df.iloc[0, :41].astype(float).tolist()
))

with open("sample_request.json", "w") as f:
    json.dump(sample, f, indent=2)
