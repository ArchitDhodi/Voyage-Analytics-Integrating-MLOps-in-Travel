"""
Train a lightweight hotel recommender (TF-IDF + cosine via k-NN) and save to models/trained/recommender.pkl.
Avoids precomputing a dense similarity matrix to keep memory small.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def main():
    root = Path(__file__).resolve().parent.parent
    hotels_path = root / "data" / "hotels.csv"
    if not hotels_path.exists():
        raise FileNotFoundError("No hotels.csv found in data/ (sample data is not used here)")

    hotels_df = pd.read_csv(hotels_path)
    # Basic dedupe on name/place to reduce size
    hotels_df = hotels_df.drop_duplicates(subset=["name", "place"], keep="first")
    for col in ["name", "place", "date"]:
        if col not in hotels_df.columns:
            hotels_df[col] = ""
    hotels_df["text"] = hotels_df[["name", "place", "date"]].fillna("").astype(str).agg(" ".join, axis=1)

    # Smaller TF-IDF to control memory
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        min_df=3,
        ngram_range=(1, 1),
    )
    tfidf = vec.fit_transform(hotels_df["text"])

    # k-NN with cosine distance; query top K at runtime instead of storing full matrix
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(tfidf)

    rec_path = root / "models" / "trained" / "recommender.pkl"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "tfidf": tfidf, "knn": knn, "hotels": hotels_df}, rec_path)
    print(f"Saved lightweight recommender to {rec_path}")


if __name__ == "__main__":
    main()
