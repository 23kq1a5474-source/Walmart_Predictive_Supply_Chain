import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

dates = pd.date_range(
    start="2024-01-01",
    end="2024-12-31",
    freq="D"
)

stores = {
    "S001": (0, 0),
    "S002": (10, 4),
    "S003": (4, 11),
    "S004": (13, 12),
    "S005": (8, 18)
}

products = {}

for i in range(1, 21):
    product_id = f"P{i:03d}"

    products[product_id] = {
        "category": ["Grocery", "Electronics", "Home", "Personal Care"][i % 4],
        "base_demand": np.random.randint(18, 80),
        "price": round(np.random.uniform(80, 1500), 2)
    }

rows = []

for date in dates:

    day_of_week = date.dayofweek
    month = date.month

    weekend_effect = 1.18 if day_of_week >= 5 else 1.0

    seasonal_effect = (
        1 + 0.15 * np.sin(2 * np.pi * date.dayofyear / 365)
    )

    holiday = int(
        (month == 1 and date.day == 26)
        or (month == 8 and date.day == 15)
        or (month == 10 and date.day in [2, 31])
        or (month == 12 and date.day in [24, 25, 31])
    )

    for store_id, (x, y) in stores.items():

        store_factor = {
            "S001": 1.00,
            "S002": 1.18,
            "S003": 0.88,
            "S004": 1.28,
            "S005": 1.08
        }[store_id]

        weather_index = np.random.normal(0, 1)

        for product_id, product in products.items():

            promotion = int(np.random.random() < 0.12)

            promotion_effect = 1.30 if promotion else 1.0

            holiday_effect = 1.25 if holiday else 1.0

            weather_effect = (
                1 + max(weather_index, -0.5) * 0.025
            )

            trend = (
                1 + 0.0007 * (date - dates[0]).days
            )

            expected_demand = (
                product["base_demand"]
                * store_factor
                * weekend_effect
                * seasonal_effect
                * promotion_effect
                * holiday_effect
                * weather_effect
                * trend
            )

            demand = max(
                0,
                int(np.random.poisson(expected_demand))
            )

            inventory = max(
                0,
                int(
                    expected_demand
                    * np.random.uniform(0.7, 1.6)
                )
            )

            supplier_delay = max(
                0,
                int(
                    np.random.choice(
                        [0, 1, 2, 3, 5],
                        p=[0.70, 0.15, 0.08, 0.05, 0.02]
                    )
                )
            )

            rows.append([
                date,
                store_id,
                product_id,
                product["category"],
                product["price"],
                promotion,
                holiday,
                weather_index,
                inventory,
                supplier_delay,
                demand,
                x,
                y
            ])


df = pd.DataFrame(
    rows,
    columns=[
        "Date",
        "Store_ID",
        "Product_ID",
        "Category",
        "Price",
        "Promotion",
        "Holiday",
        "Weather_Index",
        "Inventory",
        "Supplier_Delay_Days",
        "Demand",
        "X",
        "Y"
    ]
)

output_file = DATA_DIR / "supply_chain_data.csv"

df.to_csv(output_file, index=False)

print("Dataset created successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nFile saved at:")
print(output_file)git init