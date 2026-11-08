# Molecular Biodegradability Prediction
Predicting whether or not compounds are readily biodegradable using a machine learning algorithm on molecular descriptors.


This is an end-to-end binary classification project developed for the [DataTalks.Club Machine Learning Zoomcamp (2026)](https://github.com/DataTalksClub/machine-learning-zoomcamp)

# Problem Statement & Motivation
Biodegradability of organic compounds is an important consideration in evnrionmental screening and chemical design. Quantitative structure-activity relationship (QSAR) models are regression or classification models that use measurable/computable features of a molecule's structure to predict properties such as biodegradability. 

The goal of this small project is to ask: can a machine-learning QSAR model using numerical molecular descriptors distinguish readily biodegradable (RB) from non-readily degradable (NRB) compounds?

Our end product to address this task is a reproducible ML workflow and a working inference service. We will **not** attempt a regulatory or experimentally validated biodegradability assessment.


# Dataset
We use the UCI QSAR Biodegradation dataset [1], available [here](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation). This dataset contains

- 1055 original observations and 41 numerical molecular descriptors per compound
- The two target classes mentioned above: RB (readily biodegradable) and NRB (not readily biodegradable).
- 356 RB and 699 NRM observations in the original dataset (approximately 34% versus 66%)
- There are no missing values in the dataset, but there are three exact duplicate rows, which we remove. This leaves 1052 records we use for modeling.

The molecular descriptors include atom and bond counts, structural indicators, and topological/spectral properties of molecular graphs. Examples are `nO` (oxygen atom count), `nX` (halogen atom count), and `SpMax_L` (a spectral descriptor derived from a molecular graph Laplacian).

The source CSV is semicolon-separated and has no header row. The project loads descriptor names in the documented order and normalizes special characters in some names for XGBoost compatibility.

[1] Mansouri et al. (2013), QSAR Biodegradation, UCI Machine Learning Repository. DOI: 10.24432/C5H60M


## Exploratory analysis
The EDA notebook (`notebooks/EDA.ipynb`) investigates the class imbalance above, feature types and distributions, correlations between descriptors, and univariate relationships with the target variable. Our key observations are

- **Molecular topology is informative**: several spectral descriptors related to molecular topology have relatively strong negative correlations with the RB label.
- **Redundant descriptors**: Strong pairwise correlations appear among several graph/spectral descriptors, motivating regularization and care when interpreting feature importance
- **Non-linear associations**: The `SdO` descriptor has a weak Pearson correlation with the target (approximately `r = 0.054`) but higher estimated mutual information (approximately `0.102`). Binned observations suggest a non-monotonic association with the prevalence of RB compounds.

These observations motivated comparing linear and nonlinear classifiers. They should not be interpreted as casual evidence about chemical biodegradation pathways. 

## Model development and evaluation
After removing duplicte data, the data was split into 60% training, 20% validation, and 20% test using stratification and `random_state=42`. We tuned models on the validation set, selected a final model using the **ROC-AUC** metric, and then retrained the model on the combined training & validation data before evaluating once on the held-out test set.

### Validation results

| Model | Selected hyperparameters | Validation ROC-AUC |
| --- | --- | ---: |
| **Logistic regression** | **`C=1.0`** | **0.9204** |
| Random forest | `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1` | 0.9203 |
| XGBoost | `n_estimators=50`, `max_depth=5`, `learning_rate=0.1` | 0.9148 |
| Decision tree | `max_depth=6`, `min_samples_leaf=20` | 0.8487 |

Logistic regression and random forest performed almsot identically on validation data. We selected logistic regression for our final model since it was competative and simpler. The fitted pipeline includes `StandardScaler` and `LogisticRegression(C=1.0,max_iter=2000, random_state=42)`. The scaling is learned exclusively from the model's training data.


### Final held-out test results

| Metric | Score |
| --- | ---: |
| **ROC-AUC** | **0.9040** |
| Accuracy | 0.8673 |
| Precision (RB) | 0.8308 |
| Recall (RB) | 0.7606 |
| F1-score (RB) | 0.7941 |

The classification metrics use a **0.5 probability threshold** with **RB as the positive class**. Recall below 1 means that some readily degradable compounds are classified as NRB at this threshold. The threshold was **not** tuned on the test set.

The test results above come from the notebook's held-out evaluation, not the subsequent deployment fit. For serving predictions, `train.py` retrains the selected pipeline on **all 1052 cleaned observations** and serializes it to `model.bin`


## Repository structure

```text
molecular-biodegradability-predictor/
├── data/
│   └── biodeg.csv             # Original UCI data
├── notebooks/
│   ├── EDA.ipynb              # Exploration and statistical analysis
│   └── modeling.ipynb         # Tuning and held-out evaluation
├── train.py                   # Train deployment pipeline on cleaned data
├── predict.py                 # Flask /predict and /health endpoints
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```


## Run locally

**Requirements:** Python 3.11 and [`uv`](https://docs.astral.sh/uv/). Commands are run from the repository root.

```bash
# Install the exact dependencies recorded in uv.lock
uv sync --locked

# Rebuild the final model.bin (includes the scaler and input schema)
uv run python train.py

# Start the Flask prediction API on localhost:9696
uv run python predict.py
```

With the server running, use another terminal to verify the health endpoint:

```bash
curl http://127.0.0.1:9696/health
```

Expected response:

```json
{"model_loaded":true,"status":"ok"}
```

To reproduce the modeling experiments, launch JupyterLab with `uv run jupyter lab` and open `notebooks/modeling.ipynb`. The final comparison includes XGBoost; on macOS, XGBoost may additionally require the system OpenMP library (`brew install libomp`).

## Make a prediction

`POST /predict` requires a JSON object with **all 41 molecular descriptors**, supplied as finite numerical values. The service accepts the original UCI descriptor names (e.g. `F01[N-N]`) or the normalized aliases (`F01_N-N`).

For a reproducible example, generate a request using the first observation in the original dataset:

```bash
uv run python - <<'PY'
import json
import pandas as pd
from train import ORIGINAL_FEATURE_NAMES

data = pd.read_csv("data/biodeg.csv", sep=";", header=None)
values = [float(v) for v in data.iloc[0, :41]]
request = dict(zip(ORIGINAL_FEATURE_NAMES, values))
with open("sample_request.json", "w") as f:
    json.dump(request, f, indent=2)
print("Created sample_request.json")
PY
```

Send the request:

```bash
curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Example response **format** (the numbers below are illustrative):

```json
{
  "prediction": "RB",
  "probability_RB": 0.87,
  "threshold": 0.5
}
```

Malformed requests or missing descriptors return an HTTP `400` response with a JSON error message.

## Run with Docker

Docker packages the Python environment and prediction service. The Dockerfile **trains the deployment model inside the image** using the included data file, so a local `model.bin` is not required for the build.

```bash
# Build from the repository root
docker build -t biodegradability-api .

# Run the API and expose it on the local machine
docker run --rm --name biodeg-api \
  -p 127.0.0.1:9696:9696 biodegradability-api
```

In another terminal:

```bash
curl http://127.0.0.1:9696/health

curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

The container serves Flask using Gunicorn on port `9696`. To stop it, press `Ctrl+C` in the terminal where `docker run` is active.

## Limitations and next steps

- The dataset is relatively small, and the reported performance is based on **one stratified train/validation/test split**, not repeated cross-validation or an external benchmark.
- Random splitting does not test generalization to entirely **new molecular scaffolds or chemical families**. A scaffold-aware split would provide a stronger assessment for practical chemistry applications.
- Correlations, mutual information, and model importance scores are **associations**, not causal explanations.
- The API requires **precomputed molecular descriptors**; it does not accept molecular structures or SMILES strings and calculate descriptors automatically.
- Predictions are a research/demo screening tool, **not a substitute for experimental biodegradation testing or regulatory assessment**.

Potential future improvements include scaffold-based validation, calibration/threshold analysis, uncertainty estimation, and accepting SMILES strings through a molecular-descriptor generation pipeline.

## Acknowledgments

Developed as a midterm project for [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club. Dataset provided by the UCI Machine Learning Repository and its original authors, cited above.