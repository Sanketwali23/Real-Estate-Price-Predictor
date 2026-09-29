import pickle
import pandas as pd

with open("models/house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

test_data = pd.DataFrame([
    {
        "area_type": "Super built-up  Area",
        "location": "BTM 2nd Stage",
        "total_sqft": 1500,
        "bath": 2,
        "balcony": 1,
        "bhk": 3
    },
    {
        "area_type": "Super built-up  Area",
        "location": "BTM 2nd Stage",
        "total_sqft": 2000,
        "bath": 3,
        "balcony": 2,
        "bhk": 4
    }
])

predictions = model.predict(test_data)

for i, prediction in enumerate(predictions):
    print(
        f"Property {i + 1}: "
        f"₹{prediction:.2f} Lakhs "
        f"(₹{prediction / 100:.2f} Crore)"
    )
