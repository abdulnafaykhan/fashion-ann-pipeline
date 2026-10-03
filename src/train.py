import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

os.makedirs("models", exist_ok=True)

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

print(f"[train] Building ANN (dense_units={params['dense_units']}, "
      f"dropout={params['dropout_rate']}, lr={params['learning_rate']})...")

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(params["dense_units"], activation="relu"),
    layers.Dropout(params["dropout_rate"]),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"[train] Training for {params['epochs']} epochs (batch_size={params['batch_size']})...")
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
)

model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)

print(f"[train] Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("[train] Model saved -> models/model.h5")
print("[train] History saved -> models/history.csv")