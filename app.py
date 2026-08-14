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
# 4. TITLE
# ============================================================

st.title("🏠 Bengaluru House Price Predictor")

st.write(
    "Predict the estimated price of a property using machine learning."
)

st.divider()


# ============================================================
# 5. PROPERTY DETAILS
# ============================================================

st.subheader("🏡 Property Details")

area_type = st.selectbox(
    "Area Type",
    sorted(df["area_type"].unique())
)

location = st.selectbox(
    "Location",
    sorted(df["location"].unique())
)

total_sqft = st.number_input(
    "Total Sqft",
    min_value=300.0,
    max_value=50000.0,
    value=1000.0,
    step=50.0
)

bath = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

balcony = st.number_input(
    "Balconies",
    min_value=0,
    max_value=3,
    value=1,
    step=1
)

bhk = st.number_input(
    "BHK",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

st.divider()


# ============================================================
# 6. PREDICTION
# ============================================================

if st.button("🔮 Predict Price", use_container_width=True):

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

    st.subheader("💰 Estimated Property Price")

    st.success(
        f"₹{prediction:.2f} Lakhs"
    )

    if prediction >= 100:
        crores = prediction / 100

        st.info(
            f"Approximately ₹{crores:.2f} Crore"
        )

    st.write(
        f"**{bhk} BHK | "
        f"{total_sqft:,.0f} sqft | "
        f"{bath} Bathrooms | "
        f"{location}**"
    )