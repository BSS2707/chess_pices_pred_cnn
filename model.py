
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "chess_piece_model.keras"
CLASS_NAMES_PATH = "class_names.json"

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

    images = []
    labels = []

    class_names = sorted([
        x for x in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, x))
    ])

    for label, class_name in enumerate(class_names):

        folder = os.path.join(DATASET_DIR, class_name)

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(IMG_SIZE)

                images.append(np.array(img))
                labels.append(label)

            except Exception:
                pass

    return (
        np.array(images, dtype=np.float32),
        np.array(labels),
        class_names
    )


def build_model(num_classes):

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    x = layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    )(inputs)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def train_model(epochs=EPOCHS):

    print("Loading Dataset...")

    X, y, class_names = load_dataset()

    print("Classes:", class_names)
    print("Total Images:", len(X))

    if len(X) == 0:
        raise RuntimeError(
            "No images found inside the dataset folder."
        )

    with open(
        CLASS_NAMES_PATH,
        "w"
    ) as f:
        json.dump(
            class_names,
            f,
            indent=4
        )

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(len(X) * 0.85)

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    model, base_model = build_model(
        len(class_names)
    )

    augmentation = tf.keras.Sequential([
        layers.RandomFlip(
            "horizontal"
        ),
        layers.RandomRotation(
            0.2
        ),
        layers.RandomZoom(
            0.2
        ),
        layers.RandomContrast(
            0.2
        ),
        layers.RandomTranslation(
            0.1,
            0.1
        )
    ])

    X_train_aug = augmentation(
        X_train,
        training=True
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )

    ]

    print("Stage 1 Training...")

    model.fit(
        X_train_aug,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Stage 2 Fine Tuning...")

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    X_train_aug = augmentation(
        X_train,
        training=True
    )

    model.fit(
        X_train_aug,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=5,
        batch_size=BATCH_SIZE
    )

    model.save(
        MODEL_SAVE_PATH
    )

    print("Model Saved Successfully!")

    return model, class_names


def load_trained_model():

    if not os.path.exists(
        MODEL_SAVE_PATH
    ):
        return None, None

    if not os.path.exists(
        CLASS_NAMES_PATH
    ):
        return None, None

    try:

        model = tf.keras.models.load_model(
            MODEL_SAVE_PATH,
            compile=False
        )

    except Exception as e:

        raise RuntimeError(
            "Model loading failed.\n\n"
            "Delete the old chess_piece_model.keras file "
            "and retrain the model.\n\n"
            "Original error:\n" + str(e)
        )

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as f:

        class_names = json.load(f)

    return model, class_names


def predict_piece(
    image_input,
    model=None,
    class_names=None
):

    if model is None:

        model, class_names = load_trained_model()

    img = image_input.convert(
        "RGB"
    )

    img = img.resize(
        IMG_SIZE
    )

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    index = np.argmax(
        predictions
    )

    piece = class_names[index]

    confidence = (
        predictions[index] * 100
    )

    meta = PIECE_METADATA.get(
        piece,
        {
            "emoji": "♟",
            "display": piece
        }
    )

    probabilities = {}

    for i, name in enumerate(
        class_names
    ):

        probabilities[name] = float(
            predictions[i] * 100
        )

    return {
        "class_name": piece,
        "emoji": meta["emoji"],
        "display_name": meta["display"],
        "acc_percentage": float(
            confidence
        ),
        "formatted_output":
            f"{meta['display']} Acc {confidence:.1f}%",
        "probabilities": probabilities
    }


if __name__ == "__main__":

    train_model()
