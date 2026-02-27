import base64
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import time


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "skin_disease_model.keras",
    BASE_DIR / "skin_disease_model.h5",
    BASE_DIR / "best_skin_model.keras",
]
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"
IMG_SIZE = (224, 224)
MODEL_DISPLAY_NAME = "EfficientNetB0"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB uploads


@tf.keras.utils.register_keras_serializable()
class TrueDivide(tf.keras.layers.Layer):
    """Recreates the lambda layer exported in the H5 model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_denominator = None

    def __call__(self, *args, **kwargs):
        if not args:
            raise ValueError("TrueDivide expects at least one positional argument.")
        numerator = args[0]
        if len(args) > 1:
            denom_value = args[1]
        else:
            denom_value = kwargs.pop("denominator", 127.5)
        self._last_denominator = denom_value
        return super().__call__(numerator, **kwargs)

    def call(self, inputs, **kwargs):
        denominator = self._last_denominator
        if denominator is None:
            denominator = 127.5
        denominator = tf.cast(denominator, inputs.dtype)
        return tf.math.divide(inputs, denominator)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class Subtract(tf.keras.layers.Layer):
    """Handles subtracting a scalar constant captured during serialization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_subtrahend = None

    def __call__(self, *args, **kwargs):
        if not args:
            raise ValueError("Subtract expects at least one positional argument.")
        minuend = args[0]
        if len(args) > 1:
            subtrahend = args[1]
        else:
            subtrahend = kwargs.pop("subtrahend", 0.0)
        self._last_subtrahend = subtrahend
        return super().__call__(minuend, **kwargs)

    def call(self, inputs, **kwargs):
        subtrahend = self._last_subtrahend
        if subtrahend is None:
            subtrahend = 0.0
        subtrahend = tf.cast(subtrahend, inputs.dtype)
        return tf.math.subtract(inputs, subtrahend)

    def get_config(self):
        return super().get_config()


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p.name) for p in MODEL_CANDIDATES)
    raise FileNotFoundError(f"No model file found. Looked for: {searched}")


def load_model() -> tf.keras.Model:
    model_path = resolve_model_path()
    custom_objects = {"TrueDivide": TrueDivide, "Subtract": Subtract}
    return tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
    )


def load_class_names() -> List[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"class_names.txt not found at {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError("class_names.txt is empty.")
    return names


model = load_model()
class_names = load_class_names()


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return array


def encode_preview(image: Image.Image) -> str:
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    preview_data: Optional[str] = None
    top_predictions = []
    inference_ms: Optional[float] = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file part in the request."
        else:
            file = request.files["image"]
            if file.filename == "":
                error = "No file selected."
            else:
                try:
                    image = Image.open(file.stream)
                    preview_data = encode_preview(image)
                    processed = preprocess_image(image)
                    start = time.perf_counter()
                    preds = model.predict(processed)
                    inference_ms = (time.perf_counter() - start) * 1000
                    top_idx = np.argmax(preds[0])
                    prediction = class_names[top_idx]
                    confidence = float(preds[0][top_idx])
                    ranked_indices = np.argsort(preds[0])[::-1][:5]
                    top_predictions = [
                        {
                            "label": class_names[idx],
                            "score": float(preds[0][idx]),
                        }
                        for idx in ranked_indices
                    ]
                except Exception as exc:
                    error = f"Failed to process image: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        class_count=len(class_names),
        preview_data=preview_data,
        top_predictions=top_predictions,
        inference_ms=inference_ms,
        model_name=MODEL_DISPLAY_NAME,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
