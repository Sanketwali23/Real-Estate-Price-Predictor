import pandas as pd

df = pd.read_csv("data/Bengaluru_House_Data.csv")

# print(df.head())
# print("Shape:", df.shape)
# print("Columns:", df.columns.tolist())
# print("\nData Types:")
# print(df.dtypes)
# print("\nMissing Values:")
# print(df.isnull().sum())
# print("\nDuplicate Rows:")
# print(df.duplicated().sum())
# print("\nNumerical Summary:")
# print(df.describe())
# print("\nArea Types:")
# print(df["area_type"].value_counts())

# print("\nAvailability:")
# print(df["availability"].value_counts().head(10))

# print("\nTop Locations:")
# print(df["location"].value_counts().head(10))

# print("\nSize:")
# print(df["size"].value_counts().head(10))
# print(df["area_type"].value_counts())
# print("\nAvailability:")
# print(df["availability"].value_counts().head(15))

# print("\nTop 15 Locations:")
# print(df["location"].value_counts().head(15))

# print("\nTop 15 Societies:")
# print(df["society"].value_counts().head(15))
# Create a copy for cleaning
clean_df = df.copy()

print("Original shape:", df.shape)
print("Working copy shape:", clean_df.shape)