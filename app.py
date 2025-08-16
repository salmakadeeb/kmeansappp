import streamlit as st
import numpy as np
import pickle

# --- Page Config ---
st.set_page_config(page_title="Customer Segment Predictor", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        h1 {
            color: #1E3A8A; /* Deep Blue */
            text-align: center;
            font-size: 2.2em;
        }
        .stButton>button {
            background-color: #2563EB;
            color: white;
            font-size: 16px;
            border-radius: 12px;
            padding: 10px 24px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1D4ED8;
            transform: scale(1.05);
        }
        .stSuccess {
            background-color: #DBEAFE;
            color: #1E40AF;
            border: 2px solid #1E3A8A;
            border-radius: 12px;
            padding: 15px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load scaler & model ---
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

cluster_description = {
    0: "Supermarket-like (High Milk, Grocery, Detergents)",
    1: "Fresh Lovers (High Fresh, low other categories)",
    2: "HoReCa - Hotels/Restaurants (High Fresh + Frozen)",
    3: "Processed Food Buyers (Milk + Grocery + Detergents)"
}

# --- Layout ---
st.title("🛒 Customer Segment Predictor")
st.markdown("Enter consumer data below to discover the **customer group**.")

with st.form("input_form"):
    fresh = st.number_input("Fresh", min_value=0, value=1000)
    milk = st.number_input("Milk", min_value=0, value=1000)
    grocery = st.number_input("Grocery", min_value=0, value=1000)
    frozen = st.number_input("Frozen", min_value=0, value=500)
    detergents = st.number_input("Detergents_Paper", min_value=0, value=500)
    delicassen = st.number_input("Delicatessen", min_value=0, value=200)
    submitted = st.form_submit_button("🔍 Predict Cluster")

if submitted:
    X_new = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
    X_new_log = np.log1p(X_new)  # log transform
    X_scaled = scaler.transform(X_new_log)  # scaling
    cluster = kmeans.predict(X_scaled)[0]  # predict
    desc = cluster_description.get(cluster, "Unknown cluster")

    st.success(f"### ✅ Cluster {cluster}: {desc}")
