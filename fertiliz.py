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