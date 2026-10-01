import pandas as pd
import numpy as np

from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DATA_FILE = Path("data/supply_chain_data.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Use the most recent 30 days
# --------------------------------------------------

max_date = df["Date"].max()

start_date = max_date - pd.Timedelta(days=29)

recent_data = df[
    df["Date"] >= start_date
].copy()

print("\nInventory analysis period:")
print(
    recent_data["Date"].min(),
    "to",
    recent_data["Date"].max()
)

print(
    "Rows used:",
    len(recent_data)
)


# --------------------------------------------------
# 4. Calculate demand and inventory statistics
# --------------------------------------------------

inventory_summary = (
    recent_data
    .groupby(
        ["Store_ID", "Product_ID", "Category"]
    )
    .agg(
        Average_Daily_Demand=("Demand", "mean"),
        Demand_Std=("Demand", "std"),
        Current_Inventory=("Inventory", "mean"),
        Average_Supplier_Delay=(
            "Supplier_Delay_Days",
            "mean"
        )
    )
    .reset_index()
)


# --------------------------------------------------
# 5. Handle missing standard deviations
# --------------------------------------------------

inventory_summary["Demand_Std"] = (
    inventory_summary["Demand_Std"]
    .fillna(0)
)


# --------------------------------------------------
# 6. Inventory assumptions
# --------------------------------------------------

SERVICE_LEVEL_Z = 1.65

LEAD_TIME = 7

REVIEW_PERIOD = 14


# --------------------------------------------------
# 7. Calculate Safety Stock
# --------------------------------------------------

inventory_summary["Safety_Stock"] = (
    SERVICE_LEVEL_Z
    * inventory_summary["Demand_Std"]
    * np.sqrt(LEAD_TIME)
)


# --------------------------------------------------
# 8. Calculate Reorder Point
# --------------------------------------------------

inventory_summary["Reorder_Point"] = (
    inventory_summary["Average_Daily_Demand"]
    * LEAD_TIME
    + inventory_summary["Safety_Stock"]
)


# --------------------------------------------------
# 9. Calculate Target Inventory
# --------------------------------------------------

inventory_summary["Target_Inventory"] = (
    inventory_summary["Average_Daily_Demand"]
    * REVIEW_PERIOD
    + inventory_summary["Safety_Stock"]
)


# --------------------------------------------------
# 10. Calculate recommended order quantity
# --------------------------------------------------

inventory_summary["Recommended_Order_Qty"] = (
    inventory_summary["Target_Inventory"]
    - inventory_summary["Current_Inventory"]
).clip(lower=0)


# --------------------------------------------------
# 11. Determine whether reorder is needed
# --------------------------------------------------

inventory_summary["Reorder_Needed"] = np.where(
    inventory_summary["Current_Inventory"]
    <= inventory_summary["Reorder_Point"],
    "YES",
    "NO"
)


# --------------------------------------------------
# 12. Round numerical values
# --------------------------------------------------

numeric_columns = [
    "Average_Daily_Demand",
    "Demand_Std",
    "Current_Inventory",
    "Average_Supplier_Delay",
    "Safety_Stock",
    "Reorder_Point",
    "Target_Inventory",
    "Recommended_Order_Qty"
]

inventory_summary[numeric_columns] = (
    inventory_summary[numeric_columns]
    .round(2)
)


# --------------------------------------------------
# 13. Sort by reorder requirement
# --------------------------------------------------

inventory_summary = inventory_summary.sort_values(
    [
        "Reorder_Needed",
        "Recommended_Order_Qty"
    ],
    ascending=[False, False]
)


# --------------------------------------------------
# 14. Save recommendations
# --------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "inventory_recommendations.csv"
)

inventory_summary.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 15. Display results
# --------------------------------------------------

print("\nInventory Optimization Completed!")

print(
    "\nTotal store-product combinations:",
    len(inventory_summary)
)

reorder_count = (
    inventory_summary["Reorder_Needed"]
    .eq("YES")
    .sum()
)

print(
    "Items requiring reorder:",
    reorder_count
)

total_order_qty = (
    inventory_summary[
        "Recommended_Order_Qty"
    ].sum()
)

print(
    "Total recommended replenishment units:",
    round(total_order_qty, 2)
)


# --------------------------------------------------
# 16. Display top recommendations
# --------------------------------------------------

print("\nTop 10 Replenishment Recommendations:")

print(
    inventory_summary[
        [
            "Store_ID",
            "Product_ID",
            "Category",
            "Current_Inventory",
            "Reorder_Point",
            "Recommended_Order_Qty",
            "Reorder_Needed"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# --------------------------------------------------
# 17. Final output
# --------------------------------------------------

print("\nFile created successfully:")

print(output_file)