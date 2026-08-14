import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_house_data.csv")

print("Original shape:", df.shape)

# Remove columns we don't need for prediction
df = df.drop(columns=["size", "society", "availability"])

# Group rare locations
location_count = df["location"].value_counts()

df["location"] = df["location"].apply(
    lambda x: "Other" if location_count[x] < 10 else x
)

print("\nColumns after feature engineering:")
print(df.columns.tolist())

print("\nShape after feature engineering:")
print(df.shape)

# Save model-ready dataset
df.to_csv("data/model_ready_data.csv", index=False)

print("\nModel-ready dataset saved successfully!")