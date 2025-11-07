from fastapi import FastAPI,Path,HTTPException,Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel,computed_field,Field
from typing import Annotated,Literal,Optional
import json
import joblib
import pandas as pd

with open('croppred.pkl', 'rb') as f:
    mod=joblib.load(f)
    
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
        pred=mod.predict(input)[0]
        return JSONResponse(status_code=200, content={"predicted category": pred})
    except Exception as e:
        # This will print the actual error to your terminal
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="Model Prediction Failed")
    
