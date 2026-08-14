# 🏠 Bengaluru Real Estate Price Predictor

A Machine Learning web application that predicts **residential property prices in Bengaluru** based on property characteristics such as location, area, BHK, bathrooms, balconies, and area type.

The project covers the complete ML workflow — from **raw data cleaning and feature engineering to model training, evaluation, and cloud deployment**.

## 🚀 Live Demo

### 👉 [Try the Live Application](https://real-estate-price-predictor23.streamlit.app/)

---

## 📌 Project Overview

Real estate prices can vary significantly depending on location, property size, configuration, and other factors.

This project uses historical Bengaluru housing data to build a machine learning model capable of estimating property prices from user-provided inputs through an interactive **Streamlit web application**.

### 🔍 Input Features

- 📍 Location
- 🏢 Area Type
- 📐 Total Square Feet
- 🛏️ BHK
- 🚿 Bathrooms
- 🌇 Balconies

---

## 🧠 Machine Learning Workflow

```text
Raw Housing Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Outlier Detection & Removal
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Model Serialization (Joblib)
        ↓
Streamlit Web Application
        ↓
Cloud Deployment
```

---

## 📊 Model Performance

The final **Random Forest Regressor** achieved:

| Metric | Result |
|---|---:|
| 🎯 R² Score | **0.694** |
| 📉 MAE | **29.19 Lakhs** |
| 📉 RMSE | **74.91 Lakhs** |

The model uses preprocessing and One-Hot Encoding through a Scikit-Learn pipeline.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming |
| 🐼 Pandas | Data cleaning & manipulation |
| 🔢 NumPy | Numerical operations |
| 🤖 Scikit-Learn | ML preprocessing & modeling |
| 🌲 Random Forest | Regression model |
| 💾 Joblib | Model serialization |
| 🎈 Streamlit | Interactive web application |
| 🐙 Git & GitHub | Version control |
| ☁️ Streamlit Cloud | Application deployment |

---

## 📂 Project Structure

```text
Real-Estate-Price-Predictor/
│
├── data/
│   ├── Bengaluru_House_Data.csv
│   ├── cleaned_house_data.csv
│   └── model_ready_data.csv
│
├── models/
│   └── house_price_model.joblib
│
├── 01_inspect_data.py
├── 02_data_cleaning.py
├── 03_feature_engineering.py
├── 04_model_training.py
├── 05_test_predictions.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sanketwali23/Real-Estate-Price-Predictor.git
```

### 2. Enter the project directory

```bash
cd Real-Estate-Price-Predictor
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

---

## ✨ Key Highlights

- ✅ Cleaned and processed real-world housing data
- ✅ Converted inconsistent property-area units into square feet
- ✅ Handled missing values and duplicate records
- ✅ Detected and removed unrealistic property outliers
- ✅ Engineered BHK and location-based pricing features
- ✅ Built an end-to-end Scikit-Learn ML pipeline
- ✅ Trained and evaluated a Random Forest regression model
- ✅ Created an interactive prediction interface with Streamlit
- ✅ Deployed the application publicly on the cloud

---

## ⚠️ Disclaimer

This application is an **educational machine learning project**. Predictions are based on historical dataset patterns and should not be considered professional real-estate valuations or financial advice.

---

## 👨‍💻 Author

**Sanket Wali**

Built as an end-to-end **Data Analytics & Machine Learning portfolio project**.

⭐ If you found this project useful, consider starring the repository!