import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from main import run_prediction


FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",
    "FamilySize", "IsAlone", "Title", "Deck", "TicketGroupSize"
]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "IsAlone", "Title", "Deck"]
NUMERICAL = [column for column in FEATURES if column not in CATEGORICAL]


def build_model(estimator, scale=False):
    preprocessing = ColumnTransformer([
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL
        ),
        (
            "numbers",
            StandardScaler() if scale else "passthrough",
            NUMERICAL
        )
    ], sparse_threshold=0)
    return make_pipeline(preprocessing, estimator)


def compare_models():
    # main.py loads and preprocesses the HANA data, calls TabPFN, and returns
    # the exact same train/test rows for comparison with the local models.
    evaluation = run_prediction(include_internal=True)
    X_train = evaluation["X_train"]
    X_test = evaluation["X_test"]
    y_train = evaluation["y_train"]
    y_test = evaluation["y_test"]
    tabpfn_predictions = evaluation["tabpfn_predictions"]

    actual = y_test.to_numpy(dtype=float)
    y_train_log = np.log1p(y_train.to_numpy(dtype=float))

    if len(tabpfn_predictions) != len(actual):
        raise ValueError("TabPFN prediction count does not match test rows.")

    models = {
        "Linear Regression": build_model(LinearRegression(), scale=True),
        "Ridge Regression": build_model(Ridge(alpha=1.0), scale=True),
        "Random Forest": build_model(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),
        "Gradient Boosting": build_model(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    }

    predictions = {"TabPFN": tabpfn_predictions}

    for name, model in models.items():
        model.fit(X_train, y_train_log)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.maximum(0, np.expm1(model.predict(X_test)))
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid predictions from {name}.")
        predictions[name] = values

    return pd.DataFrame([
        {
            "Model": name,
            "MAE": mean_absolute_error(actual, values),
            "RMSE": np.sqrt(mean_squared_error(actual, values)),
            "R²": r2_score(actual, values)
        }
        for name, values in predictions.items()
    ]).sort_values("MAE").reset_index(drop=True)


def main():
    comparison = compare_models()
    print("\nModel Comparison:")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
