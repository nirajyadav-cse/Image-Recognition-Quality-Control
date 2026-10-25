"""Optional transfer-learning trainer for Good vs Defective product images.

Expected folders:
data/train/Good, data/train/Defective
data/validation/Good, data/validation/Defective

Use many representative, correctly labelled images. Do not use the resulting
model for real manufacturing decisions until it has been independently tested.
"""
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "validation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5

for folder in [TRAIN_DIR / "Good", TRAIN_DIR / "Defective",
               VAL_DIR / "Good", VAL_DIR / "Defective"]:
    folder.mkdir(parents=True, exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False
)
class_names = train_ds.class_names
if len(class_names) != 2 or len(train_ds.file_paths) < 4 or len(val_ds.file_paths) < 2:
    raise SystemExit("Add labelled images to both Good and Defective folders in train and validation first.")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save(ROOT / "model.keras")
print(f"Saved model.keras. Folder label order: {class_names}")
print("Check class_names above before interpreting the sigmoid output; evaluate on unseen images.")
