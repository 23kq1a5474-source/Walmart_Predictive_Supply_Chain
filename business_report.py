import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DATA_FILE = Path("data/supply_chain_data.csv")

FORECAST_METRICS_FILE = Path(
    "outputs/forecast_metrics.csv"
)

INVENTORY_FILE = Path(
    "outputs/inventory_recommendations.csv"
)

ROUTE_FILE = Path(
    "outputs/optimized_delivery_route.csv"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load project outputs
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

forecast_metrics = pd.read_csv(
    FORECAST_METRICS_FILE
)

inventory = pd.read_csv(
    INVENTORY_FILE
)

route = pd.read_csv(
    ROUTE_FILE
)


# --------------------------------------------------
# 3. Demand KPIs
# --------------------------------------------------

total_demand = df["Demand"].sum()

average_daily_demand = (
    df.groupby("Date")["Demand"]
    .sum()
    .mean()
)


# --------------------------------------------------
# 4. Forecasting KPIs
# --------------------------------------------------

mae = forecast_metrics.loc[
    forecast_metrics["Metric"] == "MAE",
    "Value"
].iloc[0]

rmse = forecast_metrics.loc[
    forecast_metrics["Metric"] == "RMSE",
    "Value"
].iloc[0]

mape = forecast_metrics.loc[
    forecast_metrics["Metric"] == "MAPE",
    "Value"
].iloc[0]


# --------------------------------------------------
# 5. Inventory KPIs
# --------------------------------------------------

items_requiring_reorder = (
    inventory["Reorder_Needed"]
    .eq("YES")
    .sum()
)

recommended_replenishment = (
    inventory["Recommended_Order_Qty"]
    .sum()
)


# --------------------------------------------------
# 6. Route KPI
# --------------------------------------------------

total_route_distance = (
    route["Distance_From_Previous"]
    .sum()
)


# --------------------------------------------------
# 7. Create final KPI table
# --------------------------------------------------

kpis = pd.DataFrame({
    "KPI": [
        "Total Demand",
        "Average Daily Demand",
        "Forecast MAE",
        "Forecast RMSE",
        "Forecast MAPE (%)",
        "Items Requiring Reorder",
        "Recommended Replenishment Units",
        "Total Delivery Route Distance"
    ],

    "Value": [
        total_demand,
        average_daily_demand,
        mae,
        rmse,
        mape,
        items_requiring_reorder,
        recommended_replenishment,
        total_route_distance
    ]
})


# --------------------------------------------------
# 8. Round values
# --------------------------------------------------

kpis["Value"] = kpis["Value"].round(2)


# --------------------------------------------------
# 9. Save final business report
# --------------------------------------------------

output_file = (
    OUTPUT_DIR / "final_business_kpis.csv"
)

kpis.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 10. Display final report
# --------------------------------------------------

print("\n======================================")
print("       FINAL BUSINESS REPORT")
print("======================================")

print(
    kpis.to_string(index=False)
)

print("\n======================================")
print("Business report completed successfully!")
print("======================================")

print("\nFile created:")
print(output_file)