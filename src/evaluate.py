import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = tf.keras.models.load_model("models/model.h5")

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[evaluate] Test Loss     : {test_loss:.4f}")
print(f"[evaluate] Test Accuracy : {test_acc:.4f}")

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.savefig("models/confusion_matrix.png")
print("[evaluate] Confusion matrix saved -> models/confusion_matrix.png")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_acc)}, f, indent=2)
print("[evaluate] Metrics written -> metrics.json")