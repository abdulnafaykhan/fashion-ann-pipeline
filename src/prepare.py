import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

os.makedirs("data/raw", exist_ok=True)
print("[prepare] Downloading Fashion-MNIST via tf.keras...")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

np.save("data/raw/x_train.npy", x_train)
np.save("data/raw/y_train.npy", y_train)
np.save("data/raw/x_test.npy", x_test)
np.save("data/raw/y_test.npy", y_test)

print(f"[prepare] Saved: data/raw/x_train.npy {x_train.shape}, y_train.npy")
print(f"[prepare] Saved: data/raw/x_test.npy {x_test.shape}, y_test.npy")