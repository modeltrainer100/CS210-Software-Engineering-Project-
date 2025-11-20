import os
import io
import json
import joblib
import pickle # Required for the original debug and fertilizer loading logic
import random
from typing import Annotated, Literal, Optional

# FastAPI/Pydantic
from fastapi import FastAPI, UploadFile, File, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from fastapi.middleware.cors import CORSMiddleware

# TensorFlow/Keras for Image Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image

# Utility
import uvicorn

# ----------------------------------------------------------------------
# GLOBAL CONFIGURATION & APP INITIALIZATION
# ----------------------------------------------------------------------

# Initialize the single FastAPI app instance
app = FastAPI(
    title="Combined ML and Weather API",
    description="API for Plant Disease (Keras), Crop Prediction (Joblib), Fertilizer Prediction (Pickle), and Mock Weather Data."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Determine the directory where the script is running to find model files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# ----------------------------------------------------------------------
# 1. PLANT DISEASE MODEL (TENSORFLOW/KERAS) SETUP
# ----------------------------------------------------------------------

TARGET_SIZE = (224, 224)
NUM_CLASSES = 38
PLANT_MODEL_PATH = os.path.join(MODEL_DIR, "best_cnn.keras") # saved weights file

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("✔ GPU detected — using mixed_float16 for Plant Disease Model.")
else:
    keras.mixed_precision.set_global_policy("float32")
    print("✔ CPU mode — using float32 for Plant Disease Model.")

plant_model = None
try:
    base_model = MobileNetV2(
        input_shape=(*TARGET_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    plant_model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.6),
        Dense(NUM_CLASSES, activation="softmax", dtype="float32")
    ])

    plant_model.load_weights(PLANT_MODEL_PATH)
    print("✔ Plant Disease Model architecture rebuilt and weights loaded.")

    plant_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

except Exception as e:
    print(f"❌ Error rebuilding Plant Disease model: {e}")

PLANT_CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight",
    "Corn___healthy", "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight",
    "Grape___healthy", "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot", "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___mosaic_virus", "Tomato___healthy"
]

# ----------------------------------------------------------------------
# 2. CROP PREDICTION MODEL (JOBILB) SETUP
# ----------------------------------------------------------------------

CROP_PIPELINE_PATH = os.path.join(MODEL_DIR, 'croppred.pkl')
CROP_ENCODER_PATH = os.path.join(MODEL_DIR, 'crop_label_encoder.pkl')

CROP_MODEL_PIPELINE = None
CROP_LABEL_ENCODER = None

def load_crop_model():
    """Loads the Crop Prediction model pipeline and encoder using joblib."""
    loaded_pipeline = None
    loaded_encoder = None
    
    try:
        if not os.path.exists(CROP_PIPELINE_PATH):
            raise FileNotFoundError(f"Pipeline file '{CROP_PIPELINE_PATH}' not found")
        with open(CROP_PIPELINE_PATH, 'rb') as f:
            loaded_pipeline = joblib.load(f)
        if not hasattr(loaded_pipeline, 'predict'):
            raise AttributeError("Loaded crop pipeline does not have a 'predict' method.")
    except Exception as e:
        print(f"❌ Error loading crop pipeline: {e}")
        
    try:
        if not os.path.exists(CROP_ENCODER_PATH):
            raise FileNotFoundError(f"Encoder file '{CROP_ENCODER_PATH}' not found") 
        with open(CROP_ENCODER_PATH, 'rb') as f:
            loaded_encoder = joblib.load(f)
        if not hasattr(loaded_encoder, 'inverse_transform'):
            raise AttributeError("Loaded crop encoder does not have an 'inverse_transform' method.")
    except Exception as e:
        print(f"❌ Error loading crop encoder: {e}")

    return loaded_pipeline, loaded_encoder

CROP_MODEL_PIPELINE, CROP_LABEL_ENCODER = load_crop_model()
if CROP_MODEL_PIPELINE and CROP_LABEL_ENCODER:
    print("✔ Crop Prediction Model and Encoder loaded successfully.")
else:
    print("❌ Crop Prediction Model or Encoder failed to load.")

# Pydantic Schema for Crop Prediction Input (renamed from 'features' to avoid conflict)
class CropFeatures(BaseModel):
    N: Annotated[int, Field(..., description='nitrogen value of soil')]
    P: Annotated[int, Field(..., description='Phosphorus value of soil')]
    K: Annotated[int, Field(..., description='Pottasium value of soil')]
    temperature: Annotated[float, Field(..., description='temperature')]
    humidity: Annotated[float, Field(..., description='Relative humidity')]
    ph: Annotated[float, Field(..., description='pH level')]
    rainfall: Annotated[float, Field(..., gt=0, description='rainfall in mm')]

# ----------------------------------------------------------------------
# 3. FERTILIZER MODEL (PICKLE) SETUP
# ----------------------------------------------------------------------

FERT_PIPELINE_PATH = os.path.join(MODEL_DIR, 'randpipe.pkl')
FERT_ENCODER_PATH = os.path.join(MODEL_DIR, 'fert_label_encoder.pkl')

FERT_MODEL_PIPELINE = None
FERT_LABEL_ENCODER = None

def load_fertilizer_model():
    """Loads the Fertilizer Prediction pipeline and encoder using pickle (as requested)."""
    loaded_pipeline = None
    loaded_encoder = None
    
    try:
        if not os.path.exists(FERT_PIPELINE_PATH):
            raise FileNotFoundError(f"Pipeline file '{FERT_PIPELINE_PATH}' not found")
        with open(FERT_PIPELINE_PATH, 'rb') as f:
            loaded_pipeline = pickle.load(f)
        if not hasattr(loaded_pipeline, 'predict'):
            raise AttributeError("Loaded fertilizer pipeline does not have a 'predict' method.")
    except Exception as e:
        print(f"❌ Error loading fertilizer pipeline: {e}")

    try:
        if not os.path.exists(FERT_ENCODER_PATH):
            raise FileNotFoundError(f"Encoder file '{FERT_ENCODER_PATH}' not found") 
        with open(FERT_ENCODER_PATH, 'rb') as f:
            loaded_encoder = pickle.load(f)
        if not hasattr(loaded_encoder, 'inverse_transform'):
            raise AttributeError("Loaded fertilizer encoder does not have an 'inverse_transform' method.")
    except Exception as e:
        print(f"❌ Error loading fertilizer encoder: {e}")

    return loaded_pipeline, loaded_encoder

FERT_MODEL_PIPELINE, FERT_LABEL_ENCODER = load_fertilizer_model()
if FERT_MODEL_PIPELINE and FERT_LABEL_ENCODER:
    print("✔ Fertilizer Prediction Model and Encoder loaded successfully.")
else:
    print("❌ Fertilizer Prediction Model or Encoder failed to load.")

# Pydantic Schema for Fertilizer Prediction Input
class FertilizerFeatures(BaseModel):
    Temparature: Annotated[float, Field(..., description='Enter the temperature')]
    Humidity: Annotated[float, Field(..., description='Enter the humidity')]
    Moisture: Annotated[float, Field(..., description='Enter the soil moisture')]
    soilType: Annotated[str, Field(..., description='Enter the type of soil', examples=['Red soil'])]
    croptype: Annotated[str, Field(..., description='Enter the type of crop', examples=['Rice'])]
    nitrogen: Annotated[float, Field(..., description='Enter the nitrogen content')]
    phosphorous: Annotated[float, Field(..., description='Enter the phosphorous content')]
    potassium: Annotated[float, Field(..., description='Enter the potassium content')]


# ----------------------------------------------------------------------
# 4. MOCK WEATHER API LOGIC
# ----------------------------------------------------------------------

class WeatherData(BaseModel):
    location: str
    temperature: float
    condition: str
    humidity: float
    wind_speed: float

@app.get("/api/weather/{city_name}", response_model=WeatherData, tags=["Weather"])
def get_mock_weather(city_name: str):
    """Provides mock real-time weather data for a given city."""
    # Mock logic: Return different data based on the city name (case-insensitive)
    city_name_lower = city_name.lower()
    
    if "bengaluru" in city_name_lower or "bangalore" in city_name_lower:
        temp = 25.5 + random.uniform(-2, 2)
        condition = "Partly Cloudy"
        humidity = 65 + random.uniform(-5, 5)
    elif "new york" in city_name_lower:
        temp = 10.0 + random.uniform(-5, 5)
        condition = "Chilly and Clear"
        humidity = 55 + random.uniform(-10, 10)
    else:
        temp = 20.0 + random.uniform(-10, 10)
        condition = random.choice(["Sunny", "Rainy", "Cloudy"])
        humidity = 70 + random.uniform(-20, 20)

    return WeatherData(
        location=city_name.title(),
        temperature=round(temp, 2),
        condition=condition,
        humidity=round(humidity, 2),
        wind_speed=random.uniform(5, 25)
    )

# ----------------------------------------------------------------------
# 5. API ENDPOINTS (Consolidated)
# ----------------------------------------------------------------------

@app.get("/", tags=["Root"])
def home():
    """Basic health check and welcome message."""
    return {"message": "Combined ML API running! Use /predict/disease, /predict/crop, /predict/fertilizer, or /api/weather/{city_name}"}

# --- Plant Disease Endpoint (from original block) ---
@app.post("/predict/disease", tags=["Plant Disease Detection"])
async def predict_disease(file: UploadFile = File(...)):
    """Predicts the plant disease from an uploaded image."""
    if plant_model is None:
        raise HTTPException(status_code=503, detail="Plant Disease model is not loaded.")
        
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(TARGET_SIZE)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        preds = plant_model.predict(img_array)
        pred_idx = int(np.argmax(preds[0]))
        pred_class = PLANT_CLASS_NAMES[pred_idx]
        confidence = float(np.max(preds[0]))

        return {
            "predicted_class": pred_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        print(f"Prediction error for /predict/disease: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Crop Prediction Endpoint (from original block) ---
@app.post('/predict/crop', tags=["Crop Recommendation"])
def crop_prediction(data: CropFeatures):
    model, encoder = CROP_MODEL_PIPELINE, CROP_LABEL_ENCODER 
    
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Crop Prediction model is not loaded.")

    try:
        input_data = pd.DataFrame([{
            'N': data.N,
            'P': data.P,
            'K': data.K,
            'temperature': data.temperature,
            'humidity': data.humidity,
            'ph': data.ph,
            'rainfall': data.rainfall
        }])
        
        pred = model.predict(input_data)
        pred_decoded = encoder.inverse_transform(pred)

        # Convert to normal Python type
        if isinstance(pred_decoded, (list, np.ndarray)):
            pred_decoded = pred_decoded[0]

        return JSONResponse(
            status_code=200,
            content={'Predicted crop': pred_decoded}
        )
    
    except Exception as e:
        print(f"ERROR in /predict/crop: {e}")
        raise HTTPException(status_code=500, detail="Model Prediction Failed")


# --- Fertilizer Prediction Endpoint (from original block) ---
@app.post('/predict/fertilizer', tags=["Fertilizer Recommendation"])
def fertilizer_prediction(data: FertilizerFeatures):
    """Predicts the most suitable fertilizer based on environmental data."""
    # Using the module-level loaded models for efficiency
    model, encoder = FERT_MODEL_PIPELINE, FERT_LABEL_ENCODER 
    
    if model is None or encoder is None:
        raise HTTPException(
            status_code=500, 
            detail='Model or Encoder could not be loaded. Please check file paths and validity.'
        )
    
    df = pd.DataFrame([{
        'Temparature': data.Temparature,
        'Humidity': data.Humidity,
        'Moisture': data.Moisture,
        'Soil Type': data.soilType,
        'Crop Type': data.croptype,
        'Nitrogen': data.nitrogen,
        'Phosphorous': data.phosphorous,
        'Potassium': data.potassium
    }])
    
    try:
        pred_encoded = model.predict(df)
        pred_decoded = encoder.inverse_transform(pred_encoded)
        
        if hasattr(pred_decoded, 'tolist'):
            pred_decoded = pred_decoded.tolist()
        
        return JSONResponse(
            status_code=200, 
            content={'Predicted fertilizer': pred_decoded[0]}
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f'Prediction failed: {str(e)}'
        )

# --- Consolidated Health Check (Modified to check all three ML models) ---
@app.get("/health", tags=["Health"])
def health_check_all():
    """Checks the operational status of all three ML models."""
    disease_ready = plant_model is not None
    crop_ready = CROP_MODEL_PIPELINE is not None and CROP_LABEL_ENCODER is not None
    fert_ready = FERT_MODEL_PIPELINE is not None and FERT_LABEL_ENCODER is not None
    
    if disease_ready and crop_ready and fert_ready:
        return {"status": "healthy", "disease_model_loaded": True, "crop_model_loaded": True, "fertilizer_model_loaded": True}
    else:
        raise HTTPException(
            status_code=503, 
            detail={"status": "unhealthy", "message": "One or more models failed to load.", "disease_model_loaded": disease_ready, "crop_model_loaded": crop_ready, "fertilizer_model_loaded": fert_ready}
        )

# --- Diagnostic Endpoint (from original block) ---
@app.get("/debug", tags=["Debug"])
def debug_info():
    """Provides file and loading status for ML models."""
    current_dir = os.getcwd()
    files = os.listdir(MODEL_DIR)
    
    pkl_files = [f for f in files if f.endswith('.pkl')]
    
    model_info = {
        "current_directory": current_dir,
        "all_files": files,
        "pkl_files": pkl_files,
        "model_file_exists": os.path.exists('fert_label_encoder.pkl')
    }
    
    if os.path.exists('fert_label_encoder.pkl'):
        try:
            # Using pickle as specified in the debug snippet
            with open('fert_label_encoder.pkl', 'rb') as f:
                model = pickle.load(f)
            model_info["model_type"] = str(type(model))
            model_info["model_attributes"] = dir(model)
            model_info["has_predict"] = hasattr(model, 'predict')
        except Exception as e:
            model_info["load_error"] = str(e)
    
    return JSONResponse(content=model_info)


# ----------------------------------------------------------------------
# 6. RUN SERVER
# ----------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)