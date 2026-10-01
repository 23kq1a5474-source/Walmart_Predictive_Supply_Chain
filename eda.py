import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("data/supply_chain_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# --------------------------------------------------
# 1. Total Daily Demand
# --------------------------------------------------

daily_demand = (
    df.groupby("Date")["Demand"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 5))

plt.plot(
    daily_demand["Date"],
    daily_demand["Demand"]
)

plt.title("Total Daily Demand")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_total_daily_demand.png"
)

plt.close()

# --------------------------------------------------
# 2. Demand by Store
# --------------------------------------------------

store_demand = (
    df.groupby("Store_ID")["Demand"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

store_demand.plot(kind="bar")

plt.title("Total Demand by Store")
plt.xlabel("Store")
plt.ylabel("Demand")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_demand_by_store.png"
)

plt.close()

# --------------------------------------------------
# 3. Demand by Category
# --------------------------------------------------

category_demand = (
    df.groupby("Category")["Demand"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

category_demand.plot(kind="bar")

plt.title("Total Demand by Product Category")
plt.xlabel("Category")
plt.ylabel("Demand")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_demand_by_category.png"
)

plt.close()

# --------------------------------------------------
# 4. Weekly Demand Pattern
# --------------------------------------------------

df["Day_of_Week"] = df["Date"].dt.day_name()

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_demand = (
    df.groupby("Day_of_Week")["Demand"]
    .mean()
    .reindex(day_order)
)

plt.figure(figsize=(10, 5))

weekly_demand.plot(kind="bar")

plt.title("Average Demand by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Demand")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_weekly_pattern.png"
)

plt.close()

# --------------------------------------------------
# 5. Promotion Analysis
# --------------------------------------------------

promotion_demand = (
    df.groupby("Promotion")["Demand"]
    .mean()
)

print("\nAverage Demand:")
print(promotion_demand)

print("\nAverage Demand WITHOUT Promotion:")
print(
    promotion_demand.get(0, 0)
)

print("\nAverage Demand WITH Promotion:")
print(
    promotion_demand.get(1, 0)
)

# --------------------------------------------------
# Final Output
# --------------------------------------------------

print("\nEDA completed successfully!")

print("\nGraphs saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("01_total_daily_demand.png")
print("02_demand_by_store.png")
print("03_demand_by_category.png")
print("04_weekly_pattern.png")