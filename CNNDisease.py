from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Plant Disease Detection API (.keras Model Rebuild)")

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
TARGET_SIZE = (224, 224)
NUM_CLASSES = 38                           # << IMPORTANT: MATCH your trained model
MODEL_PATH = "best_cnn.keras"   # saved weights file

# -----------------------------
# 2. MIXED PRECISION SETUP
# -----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("✔ GPU detected — using mixed_float16")
else:
    keras.mixed_precision.set_global_policy("float32")
    print("✔ CPU mode — using float32")

# -----------------------------
# 3. REBUILD MODEL ARCHITECTURE
# -----------------------------
try:
    base_model = MobileNetV2(
        input_shape=(*TARGET_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.6),
        Dense(NUM_CLASSES, activation="softmax", dtype="float32")
    ])

    print("✔ Model architecture rebuilt successfully.")

    # -----------------------------
    # 4. LOAD TRAINED WEIGHTS
    # -----------------------------
    model.load_weights(MODEL_PATH)  # FULL weight loading
    print("✔ Weights loaded successfully from .keras file.")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

except Exception as e:
    raise RuntimeError(f"❌ Error rebuilding model: {e}")

# -----------------------------
# 5. CLASS LABELS (38 CLASSES)
# -----------------------------
CLASS_NAMES = [
    # Put your 38 class names here EXACTLY in the order used during training.
    # Example:
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", 
    "Blueberry___healthy",
    "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___mosaic_virus",
    "Tomato___healthy"
]

# -----------------------------
# 6. API ENDPOINTS
# -----------------------------
@app.get("/")
def home():
    return {"message": "Plant Disease Classification API running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(TARGET_SIZE)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        preds = model.predict(img_array)
        pred_idx = int(np.argmax(preds[0]))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(np.max(preds[0]))

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# -----------------------------
# 7. RUN SERVER
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
