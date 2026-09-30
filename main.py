import os
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()
app = FastAPI(title="Titanic Fare Prediction API")


def load_data_from_hana(host, port, user, password, schema, table):
    try:
        from hdbcli import dbapi
    except ImportError as exc:
        raise ValueError(
            "SAP HANA client is missing. Install it with: pip install hdbcli"
        ) from exc

    quoted_schema = schema.replace('"', '""')
    quoted_table = table.replace('"', '""')
    set_schema_query = f'SET SCHEMA "{quoted_schema}"'
    data_query = f'SELECT * FROM "{quoted_table}"'

    connection = None
    cursor = None
    try:
        connection = dbapi.connect(
            address=host,
            port=int(port),
            user=user,
            password=password
        )
        cursor = connection.cursor()
        cursor.execute(set_schema_query)
        cursor.execute(data_query)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        return pd.DataFrame(rows, columns=column_names)
    except (ValueError, TypeError):
        raise
    except Exception as exc:
        raise ConnectionError(f"SAP HANA query failed: {exc}") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def run_prediction():
    # Configuration
    aicore_required = [
        "AICORE_AUTH_URL", "AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET",
        "AICORE_API_URL", "TABPFN_DEPLOYMENT_ID"
    ]
    hana_required = [
        "HANA_HOST", "HANA_PORT", "HANA_USER", "HANA_PASSWORD",
        "HANA_SCHEMA", "HANA_TABLE"
    ]
    required = aicore_required + hana_required
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    auth_url, client_id, client_secret, api_url, deployment_id = [
        os.environ[key] for key in aicore_required
    ]
    resource_group = os.getenv("AICORE_RESOURCE_GROUP", "default")
    hana_host = os.environ["HANA_HOST"]
    hana_port = os.environ["HANA_PORT"]
    hana_user = os.environ["HANA_USER"]
    hana_password = os.environ["HANA_PASSWORD"]
    hana_schema = os.environ["HANA_SCHEMA"]
    hana_table = os.environ["HANA_TABLE"]

    # Load and clean data
    columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",
               "Name", "Cabin", "Ticket", "Fare"]
    raw = load_data_from_hana(
        hana_host, hana_port, hana_user, hana_password,
        hana_schema, hana_table
    )

    # Match HANA column names without depending on upper/lower case.
    hana_columns = {str(column).lower(): column for column in raw.columns}
    missing = [column for column in columns if column.lower() not in hana_columns]
    if missing:
        raise ValueError(f"Missing SAP HANA columns: {', '.join(missing)}")
    raw = raw.rename(columns={
        hana_columns[column.lower()]: column for column in columns
    })

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

    # Added for FastAPI: return the existing results as JSON.
    return {
        "original_rows": len(raw),
        "usable_rows": len(df),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "metrics": {
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))
        },
        "sample_predictions": results[
            ["Actual_Fare", "Predicted_Fare", "Absolute_Error"]
        ].head(5).to_dict(orient="records")
    }



@app.post("/predict")
def predict():
    """Load Titanic data from SAP HANA and run TabPFN regression."""
    try:
        return run_prediction()
    except requests.Timeout as exc:
        raise HTTPException(status_code=504, detail="SAP AI Core request timed out.") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="SAP AI Core request failed; check server logs.") from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
