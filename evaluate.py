import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "models/brain_mri_v2.keras"
model = load_model(MODEL_PATH, compile=False)

# -----------------------------
# LOAD VALIDATION DATA
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# PREDICTION
# -----------------------------
val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("\nClassification Report\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(val_gen.class_indices.keys())
))

# -----------------------------
# CONFUSION MATRIX (SAFE SAVE)
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=val_gen.class_indices.keys(),
    yticklabels=val_gen.class_indices.keys()
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

timestamp = time.strftime("%Y%m%d_%H%M%S")
output_path = f"outputs/confusion_matrix_{timestamp}.png"
plt.savefig(output_path)
plt.close()

print(f"\nConfusion matrix saved at: {output_path}")
