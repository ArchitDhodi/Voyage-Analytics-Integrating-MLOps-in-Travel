import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_URL = os.getenv("API_URL", "http://localhost:5000")

st.set_page_config(page_title="ABMSM Dashboard", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Price Prediction", "Data Insights", "Hotel Recommendations", "Gender", "MLflow"],
)

# Styling tweaks
st.markdown(
    """
<style>
.metric-card {
  background: #000;
  color: #fff;
  padding: 1rem;
  border-radius: 0.5rem;
  text-align: center;
  border: 1px solid #333;
}
.metric-card h3, .metric-card p { color: #fff; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)


def load_csv(name):
    candidates = [ROOT / "data" / name, ROOT / "data" / "sample" / name]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c), str(c)
    raise FileNotFoundError(f"Missing {name}")


@st.cache_data
def load_all():
    flights, fp = load_csv("flights.csv")
    users, up = load_csv("users.csv")
    hotels, hp = load_csv("hotels.csv")
    flights = flights.rename(columns={"from": "source", "to": "destination", "flightType": "flight_type"})
    return flights, users, hotels, {"flights": fp, "users": up, "hotels": hp}


def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.ok
    except requests.RequestException:
        return False


is_api_up = api_health()
st.sidebar.write(f"API: {'Connected' if is_api_up else 'Offline/Mock'}")

if page == "Home":
    st.title("ABMSM Flight Price Prediction")
    flights, users, hotels, paths = load_all()
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><h3>Flights</h3><p>{len(flights)}</p></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h3>Avg Price</h3><p>${flights['price'].mean():.2f}</p></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h3>Users</h3><p>{len(users)}</p></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h3>Hotels</h3><p>{len(hotels)}</p></div>", unsafe_allow_html=True)
    st.caption(f"Data sources: {paths}")

elif page == "Price Prediction":
    st.header("Predict Flight Price")
    flights, _, _, _ = load_all()
    sources = sorted(flights["source"].dropna().unique())
    destinations = sorted(flights["destination"].dropna().unique())
    agencies = sorted(flights["agency"].dropna().unique())
    flight_types = sorted(flights["flight_type"].dropna().unique())
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source", sources)
        flight_type = st.selectbox("Flight Type", flight_types)
        agency = st.selectbox("Agency", agencies)
    with col2:
        destination = st.selectbox("Destination", destinations)
        time_h = st.slider("Duration (hours)", 0.5, 5.0, 1.5, 0.1)
        distance = st.number_input("Distance (km)", value=500)
        date_str = st.date_input("Flight date").strftime("%Y-%m-%d")
    if st.button("Predict", use_container_width=True):
        payload = {
            "source": source,
            "destination": destination,
            "flight_type": flight_type,
            "time": time_h,
            "distance": distance,
            "agency": agency,
            "date": date_str,
        }
        if is_api_up:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()
            if "predicted_price" in resp:
                st.success(f"Predicted price: ${resp['predicted_price']:.2f}")
            else:
                st.error(resp)
        else:
            st.info("API offline, showing mock prediction.")
            st.success(f"Predicted price: ${np.random.normal(800, 150):.2f} (mock)")

elif page == "Data Insights":
    st.header("Data Insights")
    flights, users, hotels, _ = load_all()
    fig = px.box(flights, x="flight_type", y="price", color="flight_type", title="Price by flight type")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.histogram(flights, x="agency", color="flight_type", title="Agency counts")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Hotel Recommendations":
    st.header("Hotel Recommendations")
    _, _, hotels, _ = load_all()
    places = sorted(hotels["place"].dropna().unique())
    place = st.selectbox("Filter by place", places)
    max_price_default = int(hotels["price"].max()) if not hotels.empty else 500
    max_price = st.slider("Max price (per day)", 0, max_price_default, max_price_default)
    top_n = st.slider("Top N", 1, 10, 5)
    if st.button("Get recommendations"):
        filt = hotels[hotels["place"] == place].copy()
        filt = filt[filt["price"] <= max_price]
        filt = filt.drop_duplicates(subset=["name", "place"])
        cols = [c for c in ["name", "place", "price", "days", "total"] if c in filt.columns]
        st.table(filt.head(top_n)[cols])

elif page == "Gender":
    st.header("Gender Classifier")
    st.write("Predict gender using your trained model (name-based).")
    _, users_df, _, _ = load_all()
    companies = sorted(users_df["company"].dropna().unique())
    name = st.text_input("Name", value="")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    company = st.selectbox("Company", companies)
    if st.button("Classify"):
        payload = {"name": name, "age": age, "company": company}
        if is_api_up:
            resp = requests.post(f"{API_URL}/gender", json=payload, timeout=5).json()
            if "gender_pred" in resp:
                st.success(f"Predicted gender: {resp['gender_pred']}")
                if "confidence" in resp:
                    st.info(f"Confidence: {resp['confidence']:.3f}")
            else:
                st.error(resp)
        else:
            st.info("API offline, mock class shown.")
            st.success(f"Predicted gender: {'male' if np.random.rand()>0.5 else 'female'} (mock)")

elif page == "MLflow":
    st.header("Model Performance")
    model_files = {
        "Price model": ROOT / "models" / "trained" / "price_model.pkl",
        "Recommender": ROOT / "models" / "trained" / "recommender.pkl",
    }
    gender_candidates = [
        ROOT / "models" / "trained" / "gender_model.pkl",
        ROOT / "models" / "trained" / "gender_model_embeddings.pkl",
    ]
    model_files["Gender model"] = next((p for p in gender_candidates if p.exists()), gender_candidates[0])
    rows = []
    for name, path in model_files.items():
        exists = path.exists()
        size_mb = round(path.stat().st_size / (1024 * 1024), 3) if exists else 0
        rows.append({"Model": name, "Exists": exists, "Size (MB)": size_mb, "Path": str(path)})
    st.table(pd.DataFrame(rows))
    st.caption("For full run metrics, open MLflow UI if running: http://localhost:5005")
