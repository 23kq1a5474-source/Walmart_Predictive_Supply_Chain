import pandas as pd
import numpy as np

from pathlib import Path
from scipy.spatial.distance import cdist


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

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Get store coordinates
# --------------------------------------------------

stores = (
    df[
        ["Store_ID", "X", "Y"]
    ]
    .drop_duplicates("Store_ID")
    .sort_values("Store_ID")
    .reset_index(drop=True)
)

print("\nStore coordinates:")
print(stores.to_string(index=False))


# --------------------------------------------------
# 4. Create warehouse
# --------------------------------------------------

warehouse = pd.DataFrame({
    "Store_ID": ["Warehouse"],
    "X": [0],
    "Y": [0]
})


locations = pd.concat(
    [warehouse, stores],
    ignore_index=True
)


# --------------------------------------------------
# 5. Create coordinate matrix
# --------------------------------------------------

coordinates = locations[
    ["X", "Y"]
].values


# --------------------------------------------------
# 6. Calculate distances
# --------------------------------------------------

distance_matrix = cdist(
    coordinates,
    coordinates,
    metric="euclidean"
)


# --------------------------------------------------
# 7. Nearest-neighbor route optimization
# --------------------------------------------------

warehouse_index = 0

unvisited = set(
    range(1, len(locations))
)

current_location = warehouse_index

route = [current_location]

total_distance = 0


while unvisited:

    nearest_location = min(
        unvisited,
        key=lambda location:
        distance_matrix[
            current_location,
            location
        ]
    )

    distance = distance_matrix[
        current_location,
        nearest_location
    ]

    total_distance += distance

    route.append(nearest_location)

    current_location = nearest_location

    unvisited.remove(
        nearest_location
    )


# --------------------------------------------------
# 8. Return to warehouse
# --------------------------------------------------

return_distance = distance_matrix[
    current_location,
    warehouse_index
]

total_distance += return_distance

route.append(warehouse_index)


# --------------------------------------------------
# 9. Create route table
# --------------------------------------------------

route_data = []

for sequence, location_index in enumerate(route, start=1):

    location = locations.iloc[
        location_index
    ]

    route_data.append({
        "Sequence": sequence,
        "Location": location["Store_ID"],
        "X": location["X"],
        "Y": location["Y"]
    })


route_df = pd.DataFrame(
    route_data
)


# --------------------------------------------------
# 10. Calculate distance from previous location
# --------------------------------------------------

route_df["Distance_From_Previous"] = 0.0

for i in range(1, len(route_df)):

    previous_index = route[i - 1]

    current_index = route[i]

    route_df.loc[
        i,
        "Distance_From_Previous"
    ] = distance_matrix[
        previous_index,
        current_index
    ]


route_df[
    "Distance_From_Previous"
] = route_df[
    "Distance_From_Previous"
].round(2)


# --------------------------------------------------
# 11. Save optimized route
# --------------------------------------------------

output_file = (
    OUTPUT_DIR /
    "optimized_delivery_route.csv"
)

route_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 12. Display route
# --------------------------------------------------

print("\nOptimized Delivery Route")
print("------------------------")

print(
    route_df.to_string(index=False)
)

print(
    "\nTotal Route Distance:",
    round(total_distance, 2)
)

print(
    "\nNumber of Stores:",
    len(stores)
)

print(
    "\nRoute starts and ends at the warehouse."
)


# --------------------------------------------------
# 13. Final output
# --------------------------------------------------

print("\nRoute optimization completed successfully!")

print("\nFile created:")
print(output_file)