import os

import joblib
import ray
from ray.util.joblib import register_ray
from sklearn.ensemble import RandomForestRegressor


register_ray()
context = ray.init()

# read data
path = "/Users/kahnwong/Git/data/nyc-trip-data/limit=5000000"
ds = ray.data.read_parquet(path).select_columns(
    ["passenger_count", "trip_distance", "total_amount"]
)

# train
model = RandomForestRegressor()

with joblib.parallel_backend("ray"):
    model.fit(
        ds.select_columns(["passenger_count", "trip_distance"]).to_pandas(),
        ds.select_columns(["total_amount"]).to_pandas(),
    )

# save model
os.makedirs("data/model", exist_ok=True)
joblib.dump(model, "data/model/model.joblib")
