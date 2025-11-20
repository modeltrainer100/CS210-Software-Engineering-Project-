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
print("DEBUG MODEL DIR:", model_dir)
print("DEBUG PIPELINE PATH:", pipeline_path)
print("DEBUG ENCODER PATH:", encoder_path)

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
@app.post('/predict/crop')
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
            content={'Predicted crop': pred_decoded[0]}
        )
    
    except Exception as e:
        # This will print the actual error to your terminal
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="Model Prediction Failed")
    
@app.get("/health")
def health_check():
    pipeline, encoder = load_model()   # unpack the tuple

    if pipeline is None or encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model not loaded"
        )
    
    return {"status": "healthy", "model_loaded": True}
