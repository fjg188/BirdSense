import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models

DATASET_PATH = "CUB_200_2011"
IMAGES_FOLDER = os.path.join(DATASET_PATH, "images")

BB_FILE       = os.path.join(DATASET_PATH, "bounding_boxes.txt")
IMAGES_FILE   = os.path.join(DATASET_PATH, "images.txt")
LABELS_FILE   = os.path.join(DATASET_PATH, "image_class_labels.txt")
SPLIT_FILE    = os.path.join(DATASET_PATH, "train_test_split.txt")

# Hyperparameters
IMG_SIZE    = 300
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE
EPOCHS      = 20  # total epochs
FREEZE_EPOCHS = 5  # first phase

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Load the CSV files
images_df = pd.read_csv(IMAGES_FILE, sep=" ", header=None, names=["image_id","path"])
labels_df = pd.read_csv(LABELS_FILE, sep=" ", header=None, names=["image_id","label"])
split_df  = pd.read_csv(SPLIT_FILE, sep=" ", header=None, names=["image_id","is_train"])
bboxes_df = pd.read_csv(BB_FILE, sep=" ", header=None, names=["image_id","x","y","w","h"])

# Merge dataframes into one
df = (images_df
      .merge(labels_df, on="image_id")
      .merge(split_df, on="image_id")
      .merge(bboxes_df, on="image_id"))

# Create class_label (0-based)
unique_labels = sorted(df["label"].unique())
label_to_index = {label: i for i, label in enumerate(unique_labels)}
df["class_label"] = df["label"] - 1

# Use the actual count of unique classes
num_classes = len(unique_labels)

# Split train/validation sets
train_df = df[df["is_train"] == 1]
val_df   = df[df["is_train"] == 0]

print(f"Number of unique classes: {num_classes}")

# Helper: loads/crops/resizes an image
def load_image(image_path, label, x, y, w, h, training):
    path = tf.convert_to_tensor(image_path, dtype=tf.string)
    label = tf.cast(label, tf.int32)
    x = tf.cast(x, tf.int32)
    y = tf.cast(y, tf.int32)
    w = tf.cast(w, tf.int32)
    h = tf.cast(h, tf.int32)

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    img_height = tf.shape(image)[0]
    img_width  = tf.shape(image)[1]

    x = tf.minimum(x, img_width - 1)
    y = tf.minimum(y, img_height - 1)
    w = tf.minimum(w, img_width  - x)
    h = tf.minimum(h, img_height - y)

    cropped = tf.image.crop_to_bounding_box(image, y, x, h, w)
    cropped = tf.image.resize(cropped, [IMG_SIZE, IMG_SIZE])
    cropped = cropped / 255.0

    return data_augmentation(cropped) if training else cropped, label


def augment_image(image, label):
    # apply data augmentation transformations
    image = data_augmentation(image)
    return image, label


# Build tf.data.Dataset
def build_dataset(dataframe, training):
    image_paths = [os.path.join(IMAGES_FOLDER, p) for p in dataframe["path"]]
    labels      = dataframe["class_label"].values
    xs          = dataframe["x"].values
    ys          = dataframe["y"].values
    ws          = dataframe["w"].values
    hs          = dataframe["h"].values

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, xs, ys, ws, hs))
    ds = ds.shuffle(buffer_size=len(dataframe))  # Shuffle before batching
    ds = ds.map(lambda p, l, x, y, w, h: load_image(p, l, x, y, w, h, training), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = build_dataset(train_df, True)
val_ds   = build_dataset(val_df, False)

# Build the base model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs=inputs, outputs=outputs)

#learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.96, staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=====================")
print("Phase 1: Training with frozen base_model")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FREEZE_EPOCHS
)

for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=====================")
print("Phase 2: Fine-tuning entire network")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=(EPOCHS - FREEZE_EPOCHS)
)

print("\nEvaluating on validation set...")
model.evaluate(val_ds)

model.save('bird_model.h5')
print("Keras model saved as bird_model.h5")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("bird_model.h5")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("bird_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as bird_model.tflite")
print("All done!")
