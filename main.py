import os
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()

# Configuration
required = ["AICORE_AUTH_URL", "AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET",
            "AICORE_API_URL", "TABPFN_DEPLOYMENT_ID"]
missing = [key for key in required if not os.getenv(key)]
if missing:
    raise ValueError(f"Missing environment variables: {', '.join(missing)}")

auth_url, client_id, client_secret, api_url, deployment_id = [
    os.environ[key] for key in required
]
resource_group = os.getenv("AICORE_RESOURCE_GROUP", "default")

# Load and clean data
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",
           "Name", "Cabin", "Ticket", "Fare"]
raw = pd.read_csv("Titanic-Dataset.csv")
missing = [column for column in columns if column not in raw.columns]
if missing:
    raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

df = raw[columns].copy()
numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
df["Sex"] = df["Sex"].astype("string").str.strip().str.lower().map(
    {"male": 0, "female": 1}
)
df["Embarked"] = df["Embarked"].astype("string").str.strip().str.upper().map(
    {"S": 0, "C": 1, "Q": 2}
).fillna(-1)
df = df.dropna(subset=["Pclass", "Sex", "SibSp", "Parch", "Fare"])
df = df.loc[df["Fare"] >= 0].copy()
if len(df) < 20:
    raise ValueError("Too few usable rows for training and evaluation.")

# Feature engineering
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
titles = df["Name"].astype("string").str.extract(
    r",\s*([^.]*)\.", expand=False
).str.strip()
title_map = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}
df["Title"] = titles.where(titles.isin(title_map), "Rare").map(title_map)
df["Deck"] = df["Cabin"].astype("string").str.strip().str.upper().str[0].map(
    dict(zip(["A", "B", "C", "D", "E", "F", "G", "T"], range(8)))
).fillna(8).astype(int)
tickets = df["Ticket"].astype("string").str.strip().replace("", pd.NA)
df["TicketGroupSize"] = tickets.map(tickets.value_counts()).fillna(1)

# Split and fill missing ages using training data
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",
            "FamilySize", "IsAlone", "Title", "Deck", "TicketGroupSize"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features].astype(float), df["Fare"].astype(float),
    test_size=0.10, random_state=42
)
X_train, X_test = X_train.copy(), X_test.copy()
age_median = X_train["Age"].median()
if pd.isna(age_median):
    raise ValueError("Training data contains no usable Age values.")
for data in (X_train, X_test):
    data["Age"] = data["Age"].fillna(age_median)

print(f"Original rows: {len(raw)}\nUsable rows: {len(df)}"
      f"\nTraining rows: {len(X_train)}\nTesting rows: {len(X_test)}")

# Regression payload
categorical = ["Pclass", "Sex", "Embarked", "IsAlone", "Title", "Deck"]
payload = {
    "task_config": {
        "task": "regression",
        "tabpfn_config": {
            "n_estimators": 8,
            "categorical_features_indices": [features.index(c) for c in categorical],
            "random_state": 0
        },
        "predict_params": {"output_type": "median"}
    },
    "x_train": X_train.to_numpy(dtype=float).tolist(),
    "y_train": np.log1p(y_train.to_numpy(dtype=float)).tolist(),
    "x_test": X_test.to_numpy(dtype=float).tolist()
}

# Authenticate, retrieve deployment URL, and predict
with requests.Session() as session:
    response = session.post(
        f"{auth_url.rstrip('/')}/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret), timeout=60
    )
    response.raise_for_status()
    headers = {
        "Authorization": f"Bearer {response.json()['access_token']}",
        "AI-Resource-Group": resource_group
    }
    response = session.get(
        f"{api_url.rstrip('/')}/lm/deployments/{deployment_id}",
        headers=headers, timeout=60
    )
    response.raise_for_status()
    deployment_url = response.json().get("deploymentUrl")
    if not deployment_url:
        raise ValueError("Deployment URL missing. Check deployment status.")

    response = session.post(
        f"{deployment_url.rstrip('/')}/predict",
        headers=headers, json=payload, timeout=300
    )
    if not response.ok:
        print("TabPFN API error:", response.text)
    response.raise_for_status()
    output = response.json()

# Convert and validate predictions
if "prediction" not in output:
    raise ValueError("TabPFN response does not contain 'prediction'.")
log_predictions = np.asarray(output["prediction"], dtype=float).reshape(-1)

if len(log_predictions) != len(y_test):
    raise ValueError("Prediction count does not match test rows.")

if not np.isfinite(log_predictions).all():
    raise ValueError("TabPFN returned invalid prediction values.")

with np.errstate(over="ignore", invalid="ignore"):
    predicted = np.maximum(0, np.expm1(log_predictions))

if not np.isfinite(predicted).all():
    raise ValueError("Predicted fares contain invalid values.")

# Results and evaluation
actual = y_test.to_numpy(dtype=float)
results = X_test.assign(
    Actual_Fare=actual, Predicted_Fare=predicted,
    Absolute_Error=np.abs(actual - predicted)
)
baseline = X_test["Pclass"].map(
    y_train.groupby(X_train["Pclass"]).median()
).fillna(y_train.median()).to_numpy(dtype=float)

# print("\nActual vs Predicted Fare — first 20 rows:")
# print(results.head(20).round(2).to_string(index=False))
print("\nRegression metrics:")
print(f"MAE:          {mean_absolute_error(actual, predicted):.4f}")
print(f"RMSE:         {np.sqrt(mean_squared_error(actual, predicted)):.4f}")
print(f"R²:           {r2_score(actual, predicted):.4f}")
