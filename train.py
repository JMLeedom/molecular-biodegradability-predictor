########################################################
# Description
########################################################
# script to train the final QSAR biodegradability classifier
########################################################



########################################################
# Imports & Settings
########################################################
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
########################################################



########################################################
# Literal Defs
########################################################
PROJECT_DIR = Path(__file__).resolve().parent

ORIGINAL_FEATURE_NAMES = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]",
    "NssssC", "nCb-", "C%", "nCp", "nO",
    "F03[C-N]", "SdssC", "HyWi_B(m)", "LOC", "SM6_L",
    "F03[C-O]", "Me", "Mi", "nN-N", "nArNO2",
    "nCRX3", "SpPosA_B(p)", "nCIR", "B01[C-Br]",
    "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d",
    "B04[C-Br]", "SdO", "TI2_L", "nCrt",
    "C-026", "F02[C-N]", "nHDon", "SpMax_B(m)",
    "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX",
]
TARGET_COLUMN = "biodegradable"

########################################################




########################################################
# Data Functions
########################################################
def clean_feature_name(name: str) -> str:
    """Match the feature name normalization used in the modeling notebook."""
    return name.replace("[", "_").replace("]", "").replace("<", "_")

def load_clean_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series, int]:
    df = pd.read_csv(
        data_path,
        sep=";",
        header=None,
        names=ORIGINAL_FEATURE_NAMES + [TARGET_COLUMN],
    )
    if len(df.columns) != 42 or len(ORIGINAL_FEATURE_NAMES) != 41:
        raise ValueError("Expected 41 molecular descriptors and one target column")

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    if not df[TARGET_COLUMN].isin({"RB", "NRB"}).all():
        raise ValueError("Target must contain only 'RB' and 'NRB' labels")

    X = df[ORIGINAL_FEATURE_NAMES].copy()
    X = X.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features must all be finite numbers; no missing values")
    X.columns = [clean_feature_name(name) for name in ORIGINAL_FEATURE_NAMES]
    if not X.columns.is_unique:
        raise ValueError("Feature names were not unique after normalization")

    y = df[TARGET_COLUMN].map({"NRB": 0, "RB": 1})
    return X, y, duplicate_count


########################################################



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "biodeg.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "model.bin")
    args = parser.parse_args()

    X, y, duplicates = load_clean_data(args.data)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    )
    model.fit(X, y)

    # Save model *and* input schema so predictions remain consistent.
    bundle = {
        "model": model,
        "feature_names": X.columns.tolist(),
        "original_feature_names": ORIGINAL_FEATURE_NAMES,
        "positive_class": "RB",
        "threshold": 0.5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)

    print(f"Removed {duplicates} exact duplicate rows")
    print(f"Trained on {len(X)} observations with {X.shape[1]} descriptors")
    print(f"Saved trained model to {args.output}")
    print("Final held-out evaluation scores are documented in notebooks/modeling.ipynb")


if __name__ == "__main__":
    main()