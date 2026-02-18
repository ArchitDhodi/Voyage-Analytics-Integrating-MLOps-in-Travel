"""
Train a simple, reliable gender model using first name text.
Saves both gender_model_embeddings.pkl and gender_model.pkl for API compatibility.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main():
    root = Path(__file__).resolve().parent.parent
    users_path = root / "data" / "users.csv"
    if not users_path.exists():
        raise FileNotFoundError(f"users.csv not found at {users_path}")

    df = pd.read_csv(users_path)
    df = df[df["gender"].isin(["male", "female"])].copy()
    df = df[["gender", "name"]].dropna()
    df["first_name"] = (
        df["name"]
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .str.lower()
    )
    df = df[df["first_name"].str.len() > 0]

    X = df[["first_name"]]
    y = df["gender"]

    preproc = ColumnTransformer(
        [
            (
                "first_name_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_features=3000,
                ),
                "first_name",
            )
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline([("prep", preproc), ("clf", clf)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, pos_label="male")
    rec = recall_score(y_val, preds, pos_label="male")
    f1 = f1_score(y_val, preds, pos_label="male")
    print({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1})

    model_dir = root / "models" / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)
    primary = model_dir / "gender_model_embeddings.pkl"
    legacy = model_dir / "gender_model.pkl"
    joblib.dump(pipeline, primary)
    joblib.dump(pipeline, legacy)
    print(f"Saved gender models to {primary} and {legacy}")


if __name__ == "__main__":
    main()
