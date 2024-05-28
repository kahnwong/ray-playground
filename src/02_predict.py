import os
from typing import Dict

import numpy as np
import ray
from ray.util.joblib import register_ray


register_ray()

# read data
path = "/Users/kahnwong/Git/data/nyc-trip-data/limit=100000"  # 30000000"
ds = ray.data.read_parquet(path).select_columns(["passenger_count", "trip_distance"])


# predict
class ScikitLearnPredictor:
    def __init__(self):
        import joblib

        self.model = joblib.load("data/model/model.joblib")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        predictions = self.model.predict(
            np.column_stack([batch["passenger_count"], batch["trip_distance"]])
        )
        batch["output"] = predictions

        return batch


predictions = ds.map_batches(ScikitLearnPredictor, concurrency=4)
print(predictions.show(limit=1))

os.makedirs("data/output", exist_ok=True)
predictions.write_parquet("data/output/predictions.parquet")
