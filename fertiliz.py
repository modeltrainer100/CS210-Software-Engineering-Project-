from fastapi import FastAPI,Path,HTTPException,Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel,computed_field,Field
from typing import Annotated
import pandas as pd
import pickle

app=FastAPI()
class fertilizer(BaseModel):
    temperature:Annotated[float,Field(...,description='Enter the temperature')]
    humidity:Annotated[float,Field(...,description='Enter the humidity')]
    soilmoisture:Annotated[float,Field(...,description='Enter the soil moisture')]
    soiltype:Annotated[str,Field(...,description='Enter the type of soil',examples=['Red soil'])]
    croptype:Annotated[str,Field(...,description='Enter the type of crop',examples=['Rice'])]
    nitrogen:Annotated[float,Field(...,description='Enter the nitrogen content')]
    phosphorous:Annotated[float,Field(...,description='Enter the phosphorous content')]
    potassium:Annotated[float,Field(...,description='Enter the potassium content')]

def load_model():
    try:
        with open('fert_label_encoder.pkl','rb') as f:
            model=pickle.load(f)
        return model
    except FileNotFoundError:
        print("Error: Model file not found")
        # Optionally, raise an exception to prevent the server from starting
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post('/prediction')
def prediction(data:fertilizer):
    df=pd.DataFrame([{
        'temperature':data.temperature,
        'humidity':data.humidity,
        'soilmoisture':data.soilmoisture,
        'soiltype':data.soiltype,
        'croptype':data.croptype,
        'nitrogen':data.nitrogen,
        'phosphorous':data.phosphorous,
        'potassium':data.potassium
    }])
    model=load_model()
    try:
        pred=model.predict(df)
    except:
        raise HTTPException(status_code=500,detail='Prediction failed')

    return JSONResponse(status_code=200,content={'Predicted fertilizer':pred})

    


