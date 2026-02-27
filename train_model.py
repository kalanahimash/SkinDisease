import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# -------------------
# 1. Basic settings
# -------------------
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = Path(
    os.environ.get("SKIN_DATASET_DIR", PROJECT_ROOT / "SkinDisease")
)  # folder containing 'train' and 'test'
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

tf.random.set_seed(SEED)
np.random.seed(SEED)


# -------------------
# 2. Load datasets
# -------------------
def load_dataset(directory: Path, training: bool) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        os.fspath(directory),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=training,
        seed=SEED if training else None,
    )


def configure_dataset(ds: tf.data.Dataset, training: bool) -> tf.data.Dataset:
    if training:
        ds = ds.shuffle(1024, seed=SEED)
    return ds.prefetch(buffer_size=AUTOTUNE)


train_raw = load_dataset(TRAIN_DIR, training=True)
val_raw = load_dataset(VAL_DIR, training=False)

class_names = train_raw.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# Save class names for inference/web usage
with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

train_ds = configure_dataset(train_raw, training=True)
val_ds = configure_dataset(val_raw, training=False)


class SmoothedSparseCrossentropy(keras.losses.Loss):
    """Sparse CE with manual label smoothing support."""

    def __init__(self, num_classes: int, label_smoothing: float = 0.0, name: str = "smoothed_sparse_ce"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.cce = keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=label_smoothing,
        )

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.num_classes)
        return self.cce(y_true_one_hot, y_pred)

# -------------------
# 3. Data augmentation
# -------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(0.2),
    ],
    name="augmentation",
)


# -------------------
# 4. Base model (EfficientNetB0 for stronger accuracy)
# -------------------
base_model = keras.applications.EfficientNetB0(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze during initial training


# -------------------
# 5. Build final model
# -------------------
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    kernel_regularizer=keras.regularizers.l2(1e-4),
)(x)

model = keras.Model(inputs, outputs, name="skin_disease_classifier")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=SmoothedSparseCrossentropy(NUM_CLASSES, label_smoothing=0.1),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
    ],
)

model.summary()


# -------------------
# 6. Compute class weights (helps with imbalance)
# -------------------
print("Computing class weights...")
all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(labels.numpy())

all_labels = np.array(all_labels)
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(all_labels),
    y=all_labels,
)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print("Class weights:", class_weight_dict)


# -------------------
# 7. Callbacks
# -------------------
log_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        verbose=1,
        min_lr=1e-6,
    ),
    ModelCheckpoint(
        "best_skin_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    TensorBoard(log_dir=str(log_dir), histogram_freq=1),
]


# -------------------
# 8. First training stage (frozen base)
# -------------------
initial_epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    callbacks=callbacks,
    class_weight=class_weight_dict,
)

model.save("skin_disease_stage1.keras")


# -------------------
# 9. Fine-tuning EfficientNet (unfreeze top layers)
# -------------------
print("Fine-tuning EfficientNetB0...")
base_model.trainable = True
fine_tune_at = 150

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=SmoothedSparseCrossentropy(NUM_CLASSES, label_smoothing=0.05),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
    ],
)

fine_tune_epochs = 20
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    callbacks=callbacks,
    class_weight=class_weight_dict,
)


# -------------------
# 10. Save final model artifacts
# -------------------
model.save("skin_disease_finetuned.keras")
print(
    "Training finished. Best checkpoint: best_skin_model.keras | "
    "Stage1 model: skin_disease_stage1.keras | Final model: skin_disease_finetuned.keras"
)
print("TensorBoard logs saved to:", log_dir)
