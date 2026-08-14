import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ============================================================
# GLOBAL CSS — PREMIUM DARK DASHBOARD THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Base ---------- */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a1f3a 0%, #0b0e1a 45%, #0a0c14 100%);
        color: #e8ecf7;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 780px;
    }

    /* ---------- Hero ---------- */
    .hero-wrap {
        text-align: center;
        padding: 1.2rem 1rem 0.6rem 1rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #7c9bff 0%, #a78bfa 45%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #9aa4c7;
        font-size: 1.02rem;
        margin-bottom: 0.9rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(124,155,255,0.18), rgba(167,139,250,0.18));
        border: 1px solid rgba(139,163,255,0.35);
        color: #b9c5ff;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 1.6rem;
    }

    /* ---------- Stat Cards ---------- */
    .stat-card {
        background: linear-gradient(160deg, rgba(124,155,255,0.10), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.05rem 0.8rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(139,163,255,0.5);
        box-shadow: 0 12px 30px rgba(80,100,255,0.25);
    }

    .stat-icon {
        font-size: 1.4rem;
        margin-bottom: 0.25rem;
    }

    .stat-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f4f6ff;
        margin: 0.1rem 0 0.15rem 0;
    }

    .stat-label {
        font-size: 0.74rem;
        color: #8f9ac0;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
    }

    /* ---------- Section Card ---------- */
    .section-card {
        background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 1.6rem 1.5rem 1.2rem 1.5rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.30);
        backdrop-filter: blur(6px);
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #eef1fb;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #e8ecf7 !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: rgba(139,163,255,0.6) !important;
    }

    .stNumberInput input {
        background-color: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #e8ecf7 !important;
    }

    .stNumberInput input:focus {
        border-color: rgba(139,163,255,0.7) !important;
        box-shadow: 0 0 0 2px rgba(124,155,255,0.20) !important;
    }

    label {
        color: #b7c0e0 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* ---------- Predict Button ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 55%, #06b6d4 100%);
        color: white;
        font-weight: 800;
        font-size: 1.05rem;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        box-shadow: 0 10px 28px rgba(99,102,241,0.45);
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 34px rgba(139,92,246,0.55);
        filter: brightness(1.08);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ---------- Result Card ---------- */
    .result-card {
        margin-top: 1.6rem;
        text-align: center;
        padding: 1.8rem 1.4rem 1.5rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(160deg, rgba(16,185,129,0.14), rgba(6,182,212,0.06));
        border: 1px solid rgba(16,185,129,0.35);
        box-shadow: 0 0 40px rgba(16,185,129,0.18), 0 10px 30px rgba(0,0,0,0.35);
    }

    .result-label {
        color: #9ee8c9;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.4rem;
    }

    .result-price {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(90deg, #34d399 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        line-height: 1.1;
    }

    .result-crore {
        color: #b6c2e6;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
    }

    .result-chip {
        display: inline-block;
        margin-top: 0.6rem;
        padding: 0.5rem 1.1rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #dbe2ff;
        font-size: 0.88rem;
        font-weight: 600;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12162a 0%, #0b0e1a 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.3rem;
        background: linear-gradient(90deg, #7c9bff, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    section[data-testid="stSidebar"] h3 {
        color: #dbe2ff !important;
        font-size: 0.95rem !important;
    }

    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li {
        color: #a9b3d6 !important;
    }

    .sidebar-pill {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.6rem;
    }

    /* ---------- Native bordered container used for Property Details ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(div.section-title) {
        background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 20px !important;
        padding: 1.6rem 1.5rem 1.2rem 1.5rem;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.30);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🏠 About")

    st.write(
        """
        **Bengaluru House Price Predictor**

        Machine Learning application for estimating
        residential property prices in Bengaluru.
        """
    )

    st.divider()

    st.subheader("🤖 Model")
    st.write("Random Forest Regressor")

    st.subheader("📊 Performance")
    st.write("R² Score: **69.4%**")
    st.write("MAE: **₹29.19 Lakhs**")

    st.divider()

    st.caption("Built with Python • Scikit-Learn • Streamlit")

# ============================================================
# 2. LOAD MODEL AND DATA
# ============================================================

model = joblib.load("models/house_price_model.joblib")

df = pd.read_csv("data/model_ready_data.csv")


# ============================================================
# 3. CREATE LOCATION PRICE/SQFT DATA
# ============================================================

df["price_per_sqft"] = (
    df["price"] * 100000
) / df["total_sqft"]

location_price = (
    df.groupby("location")["price_per_sqft"]
    .median()
)

overall_median = df["price_per_sqft"].median()


# ============================================================
# 4. HERO / TITLE
# ============================================================

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">🏠 Bengaluru House Price Predictor</div>
        <div class="hero-subtitle">AI-powered property price estimation for Bengaluru</div>
        <div class="hero-badge">⚡ Powered by Random Forest</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Top stat cards ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-icon">🤖</div>
            <div class="stat-value">Random Forest</div>
            <div class="stat-label">ML Model</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">69.4%</div>
            <div class="stat-label">R² Score</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-icon">📉</div>
            <div class="stat-value">₹29.19 L</div>
            <div class="stat-label">MAE</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 5. PROPERTY DETAILS
# ============================================================

with st.container(border=True):
    st.markdown('<div class="section-title">🏡 Property Details</div>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        area_type = st.selectbox(
            "Area Type",
            sorted(df["area_type"].unique())
        )

    with row1_col2:
        location = st.selectbox(
            "📍 Location",
            sorted(df["location"].unique())
        )

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        total_sqft = st.number_input(
            "Total Sqft",
            min_value=300.0,
            max_value=50000.0,
            value=1000.0,
            step=50.0
        )

    with row2_col2:
        bhk = st.number_input(
            "BHK",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        bath = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

    with row3_col2:
        balcony = st.number_input(
            "Balconies",
            min_value=0,
            max_value=3,
            value=1,
            step=1
        )


# ============================================================
# 6. PREDICTION
# ============================================================

predict_clicked = st.button("🔮 Predict Property Price", use_container_width=True)

if predict_clicked:

    location_price_per_sqft = location_price.get(
        location,
        overall_median
    )

    input_data = pd.DataFrame({
        "area_type": [area_type],
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "balcony": [balcony],
        "bhk": [bhk],
        "location_price_per_sqft": [location_price_per_sqft]
    })

    prediction = model.predict(input_data)[0]

    crore_html = ""
    if prediction >= 100:
        crores = prediction / 100
        crore_html = f'<div class="result-crore">≈ ₹{crores:.2f} Crore</div>'

    result_html = (
        '<div class="result-card">'
        '<div class="result-label">💰 Estimated Property Price</div>'
        f'<div class="result-price">₹{prediction:.2f} Lakhs</div>'
        f'{crore_html}'
        f'<div class="result-chip">{bhk} BHK • {total_sqft:,.0f} sqft • {bath} Bathrooms • {location}</div>'
        '</div>'
    )

    st.markdown(result_html, unsafe_allow_html=True)