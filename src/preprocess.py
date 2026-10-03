import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    params = yaml.safe_load(f)

test_size = params["preprocess"]["test_size"]
seed = params["preprocess"]["seed"]

os.makedirs("data/processed", exist_ok=True)

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")
x_test = np.load("data/raw/x_test.npy")
y_test = np.load("data/raw/y_test.npy")

print("[preprocess] Normalizing pixels to [0,1]...")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"[preprocess] Train/Val split (seed={seed}, test_size={test_size})")
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=test_size, random_state=seed
)

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/x_val.npy", x_val)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/x_test.npy", x_test)
np.save("data/processed/y_test.npy", y_test)

print(f"[preprocess] Saved: x_train({len(x_train)}) x_val({len(x_val)}) x_test({len(x_test)}) to data/processed/")