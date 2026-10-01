import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
print("Original shape:", df.shape)


# --------------------------------------------------
# 3. Sort data
# --------------------------------------------------

df = df.sort_values(
    ["Store_ID", "Product_ID", "Date"]
).reset_index(drop=True)


# --------------------------------------------------
# 4. Create lag features
# --------------------------------------------------

group_cols = ["Store_ID", "Product_ID"]

df["Lag_1"] = (
    df.groupby(group_cols)["Demand"]
    .shift(1)
)

df["Lag_7"] = (
    df.groupby(group_cols)["Demand"]
    .shift(7)
)

df["Lag_14"] = (
    df.groupby(group_cols)["Demand"]
    .shift(14)
)

df["Lag_28"] = (
    df.groupby(group_cols)["Demand"]
    .shift(28)
)


# --------------------------------------------------
# 5. Create rolling features
# --------------------------------------------------

df["Rolling_Mean_7"] = (
    df.groupby(group_cols)["Demand"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["Rolling_Mean_28"] = (
    df.groupby(group_cols)["Demand"]
    .transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
)

df["Rolling_Std_28"] = (
    df.groupby(group_cols)["Demand"]
    .transform(
        lambda x: x.shift(1).rolling(28).std()
    )
)


# --------------------------------------------------
# 6. Calendar features
# --------------------------------------------------

df["Day_of_Week"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Day_of_Month"] = df["Date"].dt.day


# --------------------------------------------------
# 7. Remove missing rows
# --------------------------------------------------

df = df.dropna().reset_index(drop=True)

print("Shape after feature creation:", df.shape)


# --------------------------------------------------
# 8. Save original IDs before encoding
# --------------------------------------------------

id_data = df[
    ["Date", "Store_ID", "Product_ID"]
].copy()


# --------------------------------------------------
# 9. One-hot encode categorical columns
# --------------------------------------------------

df = pd.get_dummies(
    df,
    columns=[
        "Store_ID",
        "Product_ID",
        "Category"
    ],
    drop_first=False
)


# --------------------------------------------------
# 10. Time-based train/test split
# --------------------------------------------------

max_date = df["Date"].max()

test_start_date = max_date - pd.Timedelta(days=29)

train_data = df[
    df["Date"] < test_start_date
].copy()

test_data = df[
    df["Date"] >= test_start_date
].copy()

print("\nTraining period:")
print(
    train_data["Date"].min(),
    "to",
    train_data["Date"].max()
)

print("\nTesting period:")
print(
    test_data["Date"].min(),
    "to",
    test_data["Date"].max()
)


# --------------------------------------------------
# 11. Select features
# --------------------------------------------------

excluded_columns = [
    "Date",
    "Demand",
    "Weather_Index",
    "Inventory",
    "Supplier_Delay_Days",
    "Price",
    "Promotion",
    "Holiday",
    "X",
    "Y"
]

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


X_train = train_data[feature_columns]
y_train = train_data["Demand"]

X_test = test_data[feature_columns]
y_test = test_data["Demand"]


# --------------------------------------------------
# 12. Train model
# --------------------------------------------------

print("\nTraining forecasting model...")

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")


# --------------------------------------------------
# 13. Make predictions
# --------------------------------------------------

predictions = model.predict(X_test)

predictions = np.maximum(
    predictions,
    0
)


# --------------------------------------------------
# 14. Calculate metrics
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

non_zero_mask = y_test != 0

mape = np.mean(
    np.abs(
        (
            y_test[non_zero_mask]
            - predictions[non_zero_mask]
        )
        / y_test[non_zero_mask]
    )
) * 100


print("\nForecasting Results")
print("-------------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")


# --------------------------------------------------
# 15. Create forecast results
# --------------------------------------------------

test_indices = test_data.index

results = id_data.loc[
    test_indices,
    ["Date", "Store_ID", "Product_ID"]
].copy()

results["Actual_Demand"] = y_test.values

results["Predicted_Demand"] = predictions


# --------------------------------------------------
# 16. Save forecast results
# --------------------------------------------------

results.to_csv(
    OUTPUT_DIR / "forecast_results.csv",
    index=False
)


# --------------------------------------------------
# 17. Save metrics
# --------------------------------------------------

metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "MAPE"
    ],
    "Value": [
        mae,
        rmse,
        mape
    ]
})

metrics.to_csv(
    OUTPUT_DIR / "forecast_metrics.csv",
    index=False
)


# --------------------------------------------------
# 18. Actual vs Predicted graph
# --------------------------------------------------

daily_actual = (
    results.groupby("Date")["Actual_Demand"]
    .sum()
)

daily_predicted = (
    results.groupby("Date")["Predicted_Demand"]
    .sum()
)

plt.figure(figsize=(12, 5))

plt.plot(
    daily_actual.index,
    daily_actual.values,
    label="Actual Demand"
)

plt.plot(
    daily_predicted.index,
    daily_predicted.values,
    label="Predicted Demand"
)

plt.title("Actual vs Predicted Daily Demand")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_actual_vs_predicted.png"
)

plt.close()


# --------------------------------------------------
# 19. Final output
# --------------------------------------------------

print("\nForecasting completed successfully!")

print("\nFiles created:")
print("forecast_results.csv")
print("forecast_metrics.csv")
print("05_actual_vs_predicted.png")