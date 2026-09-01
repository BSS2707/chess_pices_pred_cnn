import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Resolve paths relative to script location
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_SAVE_PATH = BASE_DIR / "chess_piece_model.h5"  # Changed to .h5 format
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

PIECE_METADATA = {
    "Bishop": {"emoji": "♗", "display": "Chess Bishop"},
    "King": {"emoji": "♔", "display": "Chess King"},
    "Knight": {"emoji": "♘", "display": "Chess Knight"},
    "Pawn": {"emoji": "♙", "display": "Chess Pawn"},
    "Queen": {"emoji": "♕", "display": "Chess Queen"},
    "Rook": {"emoji": "♖", "display": "Chess Rook"},
}


def load_dataset():
    images, labels = [], []
    if not os.path.exists(DATASET_DIR):
        raise RuntimeError(f"Dataset folder not found: {DATASET_DIR}")

    class_names = sorted(
        [
            x
            for x in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, x))
        ]
    )

    if not class_names:
        raise RuntimeError("No class folders found inside dataset.")

    for label, class_name in enumerate(class_names):
        folder = os.path.join(DATASET_DIR, class_name)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                img = Image.open(path).convert("RGB").resize(IMG_SIZE)
                images.append(np.array(img))
                labels.append(label)
            except Exception:
                pass

    return (
        np.array(images, dtype=np.float32),
        np.array(labels),
        class_names,
    )


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def train_model(epochs=EPOCHS):
    print("Loading Dataset...")
    X, y, class_names = load_dataset()
    print("Classes:", class_names)
    print("Total Images:", len(X))

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * 0.85)

    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]

    model, base_model = build_model(len(class_names))

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(0.1, 0.1),
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_SAVE_PATH), monitor="val_accuracy", save_best_only=True
        ),
    ]

    print("Stage 1 Training...")
    model.fit(
        augmentation(X_train, training=True),
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    print("Stage 2 Fine Tuning...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        augmentation(X_train, training=True),
        y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=BATCH_SIZE,
    )

    # Save in standard HDF5 format
    model.save(str(MODEL_SAVE_PATH))
    print("Model Saved Successfully!")
    return model, class_names


def load_trained_model():
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(
        CLASS_NAMES_PATH
    ):
        print("Model or class names not found locally.")
        if os.path.exists(DATASET_DIR):
            return train_model()
        raise FileNotFoundError(
            f"Pretrained model '{MODEL_SAVE_PATH}' missing and no dataset available to retrain."
        )

    try:
        print("Loading existing model...")
        model = tf.keras.models.load_model(str(MODEL_SAVE_PATH), compile=False)
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        print("Existing model loaded successfully.")
        return model, class_names
    except Exception as e:
        print(f"Failed to load model from path. Error: {e}")
        if os.path.exists(DATASET_DIR):
            print("Retraining model from scratch...")
            return train_model()
        raise e


def predict_piece(image_input, model=None, class_names=None):
    if model is None:
        model, class_names = load_trained_model()

    img = image_input.convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    predictions = model.predict(img_array, verbose=0)[0]
    index = np.argmax(predictions)
    piece = class_names[index]
    confidence = predictions[index] * 100

    meta = PIECE_METADATA.get(piece, {"emoji": "♟", "display": piece})
    probabilities = {
        name: float(predictions[i] * 100) for i, name in enumerate(class_names)
    }

    return {
        "class_name": piece,
        "emoji": meta["emoji"],
        "display_name": meta["display"],
        "acc_percentage": float(confidence),
        "formatted_output": f"{meta['display']} Acc {confidence:.1f}%",
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    train_model()