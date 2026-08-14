import pandas as pd


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("data/Bengaluru_House_Data.csv")


# ============================================
# 2. CREATE WORKING COPY
# ============================================

clean_df = df.copy()

print("Original shape:", clean_df.shape)


# ============================================
# 3. REMOVE DUPLICATES
# ============================================

clean_df = clean_df.drop_duplicates()

print("After removing duplicates:", clean_df.shape)


# ============================================
# 4. HANDLE MISSING VALUES
# ============================================

# Remove rows where location is missing
clean_df = clean_df.dropna(subset=["location"])

# Remove rows where size is missing
clean_df = clean_df.dropna(subset=["size"])

# Fill missing society with "Unknown"
clean_df["society"] = clean_df["society"].fillna("Unknown")

# Fill missing bath with median
clean_df["bath"] = clean_df["bath"].fillna(clean_df["bath"].median())

# Fill missing balcony with median
clean_df["balcony"] = clean_df["balcony"].fillna(
    clean_df["balcony"].median()
)

print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())


# ============================================
# 5. CHECK total_sqft VALUES
# ============================================

print("\nSample unusual total_sqft values:")
print(
    clean_df[
        ~clean_df["total_sqft"]
        .str.replace(".", "", regex=False)
        .str.isnumeric()
    ]["total_sqft"].head(20)
)


# ============================================
# 6. CONVERT total_sqft TO SQ FT
# ============================================

def convert_sqft(value):

    value = str(value).strip()

    # Range: 2100 - 2850
    if "-" in value:
        low, high = value.split("-")
        return (float(low) + float(high)) / 2

    # Square Meter
    if "Sq. Meter" in value:
        number = float(value.replace("Sq. Meter", "").strip())
        return number * 10.7639

    # Square Yards
    if "Sq. Yards" in value:
        number = float(value.replace("Sq. Yards", "").strip())
        return number * 9

    # Perch
    if "Perch" in value:
        number = float(value.replace("Perch", "").strip())
        return number * 272.25

    # Acres
    if "Acres" in value:
        number = float(value.replace("Acres", "").strip())
        return number * 43560

    # Cents
    if "Cents" in value:
        number = float(value.replace("Cents", "").strip())
        return number * 435.6

    # Guntha
    if "Guntha" in value:
        number = float(value.replace("Guntha", "").strip())
        return number * 1089

    # Grounds
    if "Grounds" in value:
        number = float(value.replace("Grounds", "").strip())
        return number * 2400

    # If the value is a normal number
    try:
        return float(value)
    except ValueError:
        return None
def extract_bhk(value):
    return int(str(value).split()[0])

clean_df["total_sqft"] = clean_df["total_sqft"].apply(convert_sqft)

print("Conversion completed!")

print("\nUnconverted values:")
print(clean_df[clean_df["total_sqft"].isnull()])
# ============================================
# 7. CHECK RESULT
# ============================================

print("\nFirst 10 total_sqft values:")
print(clean_df["total_sqft"].head(10))

print("\ntotal_sqft data type:")
print(clean_df["total_sqft"].dtype)
clean_df["bhk"] = clean_df["size"].apply(extract_bhk)
print("\nBHK values:")
print(clean_df["bhk"].value_counts().sort_index())
print("\nProperties with BHK > 10:")
print(clean_df[clean_df["bhk"] > 10][["size", "bhk", "total_sqft", "price"]])
clean_df = clean_df[clean_df["bhk"] <= 10]
print("\nShape after BHK cleaning:", clean_df.shape)
print("\nMaximum BHK:", clean_df["bhk"].max())
print("\nBathroom distribution:")
print(clean_df["bath"].value_counts().sort_index())
print("\nProperties with bath > 10:")
print(clean_df[clean_df["bath"] > 10][["size", "bhk", "bath", "total_sqft", "price"]])

print("\nBathroom vs BHK outliers:")
print(
    clean_df[
        clean_df["bath"] > clean_df["bhk"] + 2
    ][["size", "bhk", "bath", "total_sqft", "price"]]
)

print(
    "\nNumber of bathroom outliers:",
    len(clean_df[clean_df["bath"] > clean_df["bhk"] + 2])
)
clean_df["bath_per_bhk"] = clean_df["bath"] / clean_df["bhk"]
print("\nHighest bath/BHK ratios:")
print(
    clean_df[
        ["size", "bhk", "bath", "bath_per_bhk", "total_sqft", "price"]
    ]
    .sort_values("bath_per_bhk", ascending=False)
    .head(20)
)

clean_df = clean_df[clean_df["bath"] <= 10]
clean_df = clean_df.drop(columns=["bath_per_bhk"])
print("\nMaximum bathrooms:", clean_df["bath"].max())
print("Shape after bathroom cleaning:", clean_df.shape)

clean_df["price_per_sqft"] = (clean_df["price"] * 100000) / clean_df["total_sqft"]
print("\nPrice per sqft summary:")
print(clean_df["price_per_sqft"].describe())

print("\nExtreme price per sqft:")
print(
    clean_df[
        (clean_df["price_per_sqft"] < 500) |
        (clean_df["price_per_sqft"] > 50000)
    ][["location", "bhk", "total_sqft", "price", "price_per_sqft"]].head(20)
)

print("\nSuspicious total_sqft:")
print(
    clean_df[
        (clean_df["total_sqft"] < 300) |
        (clean_df["total_sqft"] > 10000)
    ][["location", "size", "bhk", "total_sqft", "price"]].head(30)
)

print(
    "\nNumber of suspicious total_sqft:",
    len(clean_df[
        (clean_df["total_sqft"] < 300) |
        (clean_df["total_sqft"] > 10000)
    ])
)

clean_df["sqft_per_bhk"] = clean_df["total_sqft"] / clean_df["bhk"]
print("\nSuspicious sqft per BHK:")
print(
    clean_df[
        (clean_df["sqft_per_bhk"] < 300) |
        (clean_df["sqft_per_bhk"] > 5000)
    ][
        ["location", "size", "bhk", "total_sqft", "sqft_per_bhk", "price"]
    ].head(30)
)
print(
    "\nNumber of sqft/BHK outliers:",
    len(
        clean_df[
            (clean_df["sqft_per_bhk"] < 300) |
            (clean_df["sqft_per_bhk"] > 5000)
        ]
    )
)

# Remove unrealistic total_sqft
clean_df = clean_df[
    (clean_df["total_sqft"] >= 300) &
    (clean_df["total_sqft"] <= 50000)
]

# Remove unrealistic sqft per BHK
clean_df = clean_df[
    (clean_df["sqft_per_bhk"] >= 300) &
    (clean_df["sqft_per_bhk"] <= 5000)
]

print("\nShape after sqft/BHK cleaning:", clean_df.shape)

# Remove helper column
clean_df = clean_df.drop(columns=["sqft_per_bhk"])

print("\nCurrent columns:")
print(clean_df.columns.tolist())
print("\nPrice summary:")
print(clean_df["price"].describe())
print("\nHighest price properties:")
print(
    clean_df[
        ["location", "bhk", "total_sqft", "price"]
    ].sort_values("price", ascending=False).head(15)
)

print("\nPrice per sqft summary:")
print(clean_df["price_per_sqft"].describe())

print("\nHighest price per sqft:")
print(
    clean_df[
        ["location", "bhk", "total_sqft", "price", "price_per_sqft"]
    ]
    .sort_values("price_per_sqft", ascending=False)
    .head(20)
)

print("\nLowest price per sqft:")
print(
    clean_df[
        ["location", "bhk", "total_sqft", "price", "price_per_sqft"]
    ]
    .sort_values("price_per_sqft")
    .head(20)
)

clean_df = clean_df[
    (clean_df["price_per_sqft"] >= 1000) &
    (clean_df["price_per_sqft"] <= 50000)
]

print("\nShape after price cleaning:", clean_df.shape)

print("\nPrice per sqft range:")
print(clean_df["price_per_sqft"].min())
print(clean_df["price_per_sqft"].max())

clean_df = clean_df.drop(columns=["price_per_sqft"])
print("\nColumns after cleaning:")
print(clean_df.columns.tolist())
clean_df.to_csv("data/cleaned_house_data.csv", index=False)

print("\nCleaned dataset saved successfully!")