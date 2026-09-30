from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Titanic dataset
raw = pd.read_csv(Path(__file__).resolve().parent / "Titanic-Dataset.csv")
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",
           "Name", "Cabin", "Ticket", "Fare"]
df = raw[columns].copy()

# data preprocessing
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
    raise ValueError("Too few usable rows.")

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

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",
            "FamilySize", "IsAlone", "Title", "Deck", "TicketGroupSize"]
categorical = ["Pclass", "Sex", "Embarked", "IsAlone", "Title", "Deck"]
numerical = [column for column in features if column not in categorical]

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

actual = y_test.to_numpy(dtype=float)
y_train_log = np.log1p(y_train.to_numpy(dtype=float))

# Import TabPFN results; retain row IDs to verify prediction alignment
from main import predicted as tabpfn_predictions, X_test as tabpfn_test_rows

pd.testing.assert_frame_equal(X_test, tabpfn_test_rows)
if len(tabpfn_predictions) != len(actual):
    raise ValueError("TabPFN prediction count does not match test rows.")

def build_model(estimator, scale=False):
    preprocessing = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", StandardScaler() if scale else "passthrough", numerical)
    ], sparse_threshold=0)
    return make_pipeline(preprocessing, estimator)

models = {
    "Linear Regression": build_model(LinearRegression(), scale=True),
    "Ridge Regression": build_model(Ridge(alpha=1.0), scale=True),
    "Random Forest": build_model(
        RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    ),
    "Gradient Boosting": build_model(
        GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, random_state=42
        )
    )
}

if __name__ == "__main__":
    predictions = {"TabPFN": tabpfn_predictions}

    for name, model in models.items():
        model.fit(X_train, y_train_log)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.maximum(0, np.expm1(model.predict(X_test)))
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid predictions from {name}.")
        predictions[name] = values

    comparison = pd.DataFrame([
        {
            "Model": name,
            "MAE": mean_absolute_error(actual, values),
            "RMSE": np.sqrt(mean_squared_error(actual, values)),
            "R²": r2_score(actual, values)
        }
        for name, values in predictions.items()
    ]).sort_values("MAE").reset_index(drop=True)

    print("\nModel Comparison:")
    print(comparison.round(4).to_string(index=False))