import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# CONFIGURATION
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 4

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    brightness_range=[0.9, 1.1],
    horizontal_flip=True
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    "dataset/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(weights))

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS (WINDOWS SAFE)
# -----------------------------
checkpoint = callbacks.ModelCheckpoint(
    filepath="models/checkpoint_epoch_{epoch:02d}.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

# -----------------------------
# TRAINING
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# SAVE FINAL MODEL (ONLY ONCE)
# -----------------------------
FINAL_MODEL_PATH = "models/brain_mri_v2.keras"
model.save(FINAL_MODEL_PATH)
print(f"\nFinal model saved successfully to {FINAL_MODEL_PATH}")

# -----------------------------
# EVALUATION
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
plt.savefig("outputs/confusion_matrix.png")
plt.show()
