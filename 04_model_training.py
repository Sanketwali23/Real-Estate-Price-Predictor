
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/model_ready_data.csv")

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["price"])
y = df["price"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("price")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 4. CREATE LOCATION PRICE PER SQFT FEATURE
# ============================================================

train_data = X_train.copy()
train_data["price"] = y_train.values

train_data["price_per_sqft"] = (
    train_data["price"] * 100000
) / train_data["total_sqft"]

# Median price/sqft for each location
location_price = (
    train_data
    .groupby("location")["price_per_sqft"]
    .median()
)

# Overall median for unknown locations
overall_median = train_data["price_per_sqft"].median()

# Add feature to training data
X_train = X_train.copy()

X_train["location_price_per_sqft"] = (
    X_train["location"]
    .map(location_price)
    .fillna(overall_median)
)

# Add feature to testing data
X_test = X_test.copy()

X_test["location_price_per_sqft"] = (
    X_test["location"]
    .map(location_price)
    .fillna(overall_median)
)

print("\nLocation price feature added!")


# ============================================================
# 5. COLUMN TYPES
# ============================================================

categorical_columns = [
    "area_type",
    "location"
]

numerical_columns = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "location_price_per_sqft"
]


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# ============================================================
# 7. MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("regressor", RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features=0.8,
            max_depth=35
        ))
    ]
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 9. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nModel Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)


# ============================================================
# 11. SAVE MODEL
# ============================================================
joblib.dump(
    model,
    "models/house_price_model.joblib",
    compress=3
)

print("\nCompressed model saved successfully!")


# ============================================================
# 12. CHECK BTM 2ND STAGE DATA
# ============================================================

print("\nBTM 2nd Stage sample:")

btm_data = df[
    df["location"].str.contains(
        "BTM 2nd Stage",
        case=False,
        na=False
    )
][
    [
        "location",
        "total_sqft",
        "bath",
        "bhk",
        "price"
    ]
].sort_values("total_sqft")

print(btm_data.tail(20))
