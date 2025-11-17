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
    

    from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, computed_field, Field
from typing import Annotated
import pandas as pd
import pickle
import os

app = FastAPI()
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import pandas as pd
import pickle
import os

app = FastAPI()

# Diagnostic endpoint
@app.get("/debug")
def debug_info():
    current_dir = os.getcwd()
    files = os.listdir('.')
    pkl_files = [f for f in files if f.endswith('.pkl')]
    
    model_info = {
        "current_directory": current_dir,
        "all_files": files,
        "pkl_files": pkl_files,
        "model_file_exists": os.path.exists('fert_label_encoder.pkl')
    }
    
    # Try to load the model and get more details
    if os.path.exists('fert_label_encoder.pkl'):
        try:
            with open('fert_label_encoder.pkl', 'rb') as f:
                model = pickle.load(f)
            model_info["model_type"] = str(type(model))
            model_info["model_attributes"] = dir(model)
            # Check if it has predict method
            model_info["has_predict"] = hasattr(model, 'predict'    )
        except Exception as e:
            model_info["load_error"] = str(e)
    
    return JSONResponse(content=model_info)
class fertilizer(BaseModel):
    Temparature: Annotated[float, Field(..., description='Enter the temperature')]
    Humidity: Annotated[float, Field(..., description='Enter the humidity')]
    Moisture: Annotated[float, Field(..., description='Enter the soil moisture')]
    soilType: Annotated[str, Field(..., description='Enter the type of soil', examples=['Red soil'])]
    croptype: Annotated[str, Field(..., description='Enter the type of crop', examples=['Rice'])]
    nitrogen: Annotated[float, Field(..., description='Enter the nitrogen content')]
    phosphorous: Annotated[float, Field(..., description='Enter the phosphorous content')]
    potassium: Annotated[float, Field(..., description='Enter the potassium content')]

# Assuming your files are located next to your main script for easier deployment
# If you are running this from a specific directory, adjust `model_dir` if needed.
model_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
pipeline_path = os.path.join(model_dir, 'randpipe.pkl')
encoder_path = os.path.join(model_dir, 'fert_label_encoder.pkl')

def load_model():
    """
    Loads the trained ML pipeline and the fitted LabelEncoder.

    Returns:
        tuple[Pipeline, LabelEncoder] or tuple[None, None]: 
        A tuple containing the loaded pipeline and encoder, or (None, None) on failure.
    """
    loaded_pipeline = None
    loaded_encoder = None
    
    # 1. Load the Model Pipeline
    try:
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file '{pipeline_path}' not found")
            
        with open(pipeline_path, 'rb') as f:
            loaded_pipeline = pickle.load(f)
            
        if not hasattr(loaded_pipeline, 'predict'):
            raise AttributeError("Loaded pipeline does not have a 'predict' method.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return (None, None)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return (None, None)

    # 2. Load the Label Encoder
    try:
        if not os.path.exists(encoder_path):
            # Use the correct file name you mentioned in the debug endpoint
            raise FileNotFoundError(f"Encoder file '{encoder_path}' not found") 
            
        with open(encoder_path, 'rb') as f:
            loaded_encoder = pickle.load(f)
            
        if not hasattr(loaded_encoder, 'inverse_transform'):
            raise AttributeError("Loaded encoder does not have an 'inverse_transform' method.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return (None, None)
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return (None, None)

    # Return the pipeline and the encoder as a tuple
    return (loaded_pipeline, loaded_encoder)

@app.post('/prediction')
def prediction(data: fertilizer):
    # 1. Load model and encoder, unpacking the tuple
    # NOTE: You must have updated load_model() to return (pipeline, encoder)
    model, encoder = load_model()
    
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
        # 3. Predict (returns encoded number, e.g., [1])
        pred_encoded = model.predict(df)
        
        
        # 4. Decode the prediction using the loaded encoder
        pred_decoded = encoder.inverse_transform(pred_encoded)
        
        # Convert numpy array to list for JSON serialization
        if hasattr(pred_decoded, 'tolist'):
            pred_decoded = pred_decoded.tolist()
        
        # Return the single predicted string (e.g., 'Urea')
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

@app.get("/health")
def health_check():
    # Unpack the tuple returned by the modified load_model function
    model, encoder = load_model()
    
    # Check if EITHER the model (pipeline) OR the encoder is None
    if model is None or encoder is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: Model or Encoder not loaded properly"
        )
        
    return {"status": "healthy", "model_loaded": True}


from fastapi import FastAPI,Path,HTTPException,Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel,computed_field,Field
from typing import Annotated,Literal,Optional
import json
import joblib
import pandas as pd
import os
model_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
pipeline_path = os.path.join(model_dir, 'croppred.pkl')
encoder_path = os.path.join(model_dir, 'crop_label_encoder.pkl')

def load_model():
    
    loaded_pipeline = None
    loaded_encoder = None
    
    # 1. Load the Model Pipeline
    try:
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file '{pipeline_path}' not found")
            
        with open(pipeline_path, 'rb') as f:
            loaded_pipeline = joblib.load(f)
            
        if not hasattr(loaded_pipeline, 'predict'):
            raise AttributeError("Loaded pipeline does not have a 'predict' method.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return (None, None)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return (None, None)

    # 2. Load the Label Encoder
    try:
        if not os.path.exists(encoder_path):
            # Use the correct file name you mentioned in the debug endpoint
            raise FileNotFoundError(f"Encoder file '{encoder_path}' not found") 
            
        with open(encoder_path, 'rb') as f:
            loaded_encoder = joblib.load(f)
            
        if not hasattr(loaded_encoder, 'inverse_transform'):
            raise AttributeError("Loaded encoder does not have an 'inverse_transform' method.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return (None, None)
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return (None, None)

    # Return the pipeline and the encoder as a tuple
    return (loaded_pipeline, loaded_encoder)
   

    
app=FastAPI()
class features(BaseModel):
    N:Annotated[int ,Field(...,description='nitrogen value of soil')]
    P:Annotated[int,Field(...,description='Phosphorus value of soil')]
    K:Annotated[int,Field(...,description='Pottasium value of soil')]
    temperature:Annotated[float,Field(..., description='temperature')]
    humidity:Annotated[float,Field(...,description='Gender of the patient')]
    ph:Annotated[float,Field(...,description='pH level')]
    rainfall:Annotated[float,Field(...,gt=0,description='Weight of the patient in kg')]
@app.post('/crop')
def cropPediction(data: features):
    model, encoder = load_model()
    try:
        input=pd.DataFrame([{
        'N':data.N,
        'P': data.P,
        'K': data.K,
        'temperature': data.temperature,
        'humidity': data.humidity,
        'ph': data.ph,
        'rainfall': data.rainfall
        }]
        )
        pred=model.predict(input)
        pred_decoded=encoder.inverse_transform(pred)
        if hasattr(pred_decoded, 'tolist'):
            pred_decoded = pred_decoded.tolist()
        
        # Return the single predicted string (e.g., 'Urea')
        return JSONResponse(
            status_code=200, 
            content={'Predicted fertilizer': pred_decoded[0]}
        )
    
    except Exception as e:
        # This will print the actual error to your terminal
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="Model Prediction Failed")
    
@app.get("/health")
def health_check():
    model = load_model()
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable: Model not loaded"
        )
    return {"status": "healthy", "model_loaded": True}