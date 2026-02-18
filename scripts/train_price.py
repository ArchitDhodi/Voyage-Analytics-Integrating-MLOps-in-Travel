"""
Train the price model and save to models/trained/price_model.pkl.
Includes preprocessing so the API can feed raw fields.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date_parsed"] = pd.to_datetime(out["date"], errors="coerce")
    out["month"] = out["date_parsed"].dt.month.fillna(0).astype(int)
    out["dayofweek"] = out["date_parsed"].dt.dayofweek.fillna(0).astype(int)
    out["speed"] = out["distance"] / out["time"].replace(0, np.nan)
    out["speed"] = out["speed"].fillna(out["speed"].median())
    return out.drop(columns=["date_parsed"])


def main():
    root = Path(__file__).resolve().parent.parent
    data_candidates = [root / "data" / "flights.csv", root / "data" / "sample" / "flights.csv"]
    flights_path = next((p for p in data_candidates if p.exists()), None)
    if flights_path is None:
        raise FileNotFoundError("Could not find flights.csv in data/ or data/sample/")

    flights = pd.read_csv(flights_path).rename(columns={"from": "source", "to": "destination", "flightType": "flight_type"})
    flights = flights.dropna(subset=["price"])

    base_features = ["source", "destination", "flight_type", "time", "distance", "agency", "date"]

    cat_cols = ["source", "destination", "flight_type", "agency", "month", "dayofweek"]
    num_cols = ["time", "distance", "speed"]

    preproc = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", "passthrough", num_cols)],
        remainder="drop",
    )

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    price_pipe = Pipeline(
        [
            ("features", FunctionTransformer(add_features, validate=False)),
            ("prep", preproc),
            ("model", rf),
        ]
    )

    X = flights[base_features]
    y = flights["price"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    price_pipe.fit(X_train, y_train)
    preds = price_pipe.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    print({"rmse": rmse, "mae": mae, "r2": r2})

    out_path = root / "models" / "trained" / "price_model.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(price_pipe, out_path)
    print(f"Saved price model to {out_path}")


if __name__ == "__main__":
    main()
