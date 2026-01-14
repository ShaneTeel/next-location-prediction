# backend/geolife_api.py
# Libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import uvicorn
from typing import List, Dict
from geolife_model import load_geolife_model, cluster_geolife_user, model_pipe, save_model

app = FastAPI()

class ClusteringRequest(BaseModel):
    """Input for User-specific Clustering"""
    uid: object
    distance: float
    min_k: int

class ModelingRequest(BaseModel):
    """Input for User-specific Modeling"""
    uid: object
    distance: float
    min_k: int
    thresh: int
    min_samples: int

class SavingRequest(BaseModel):
    """Input for User-specific Modeling"""
    uid: object
    distance: float
    min_k: int
    thresh: int
    min_samples: int

class NextLocationPredictionRequest(BaseModel):
    """Input for User-specific Next Location Prediction"""
    uid: object
    lat_origin: float
    lng_origin: float
    timedelta: float
    month: int
    day: int
    day_of_week: int
    hour_in_day: int
    minute_in_hour: int

@app.get("/")
def read_root():
    return {"message": "GeoLife Next Location Prediction API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/cluster")
def cluster_locations(request: ClusteringRequest):

    try:
        print("attempting to cluster")
        df, scores = cluster_geolife_user(request.uid, request.distance, request.min_k)
        api_dict = {
            'df': df.to_dict(orient='records'),
            'scores': scores
        }
        return api_dict
    
    except Exception as e: 
        print("Error encountered")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/model")
def model_clusters(request: ModelingRequest):

    try:
        print("attempting to cluster")
        df, _ = cluster_geolife_user(request.uid, request.distance, request.min_k)
        print("clustering succeeded")
        print('attempting to model')
        model_scores, final_df = model_pipe(df, request.thresh, request.min_samples)
        print('modeling succeeded')
        api_dict = {
            'model_scores': model_scores,
            'df': final_df.to_dict(orient='records')
        }
        print('return dictionary')

        return api_dict
    
    except Exception as e: 
        print("Error encountered")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/save")
def saving_model(request: SavingRequest):

    try:
        print("attempting to cluster")
        df, _ = cluster_geolife_user(request.uid, request.distance, request.min_k)
        print("clustering succeeded")
        print('attempting to model')
        df = save_model(df, request.uid, request.thresh, request.min_samples)
        print('modeling succeeded')
        message = ['Model Saved']
        save_api_dict = {
            'Message': message,
            'df': df.to_dict(orient='records')
        }
        return save_api_dict
    
    except Exception as e: 
        print("Error encountered")
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_next_location(request: NextLocationPredictionRequest):
    
    try:
        print('loading model')
        model, feature_names, df = load_geolife_model(request.uid)
        print('model loaded successfully')
        print("processing features for conversion to dataframe")
        features = {
            'uid': request.uid,
            'lat_origin': request.lat_origin,
            'lng_origin': request.lng_origin,
            'timedelta': request.timedelta,
            'month': request.month,
            'day': request.day,
            'day_of_week': request.day_of_week,
            'hour_in_day': request.hour_in_day,
            'minute_in_hour': request.minute_in_hour
        }
        
        input_data = [[
            features['uid'],
            features['lat_origin'],
            features['lng_origin'],
            features['timedelta'],
            features['month'],
            features['day'],
            features['day_of_week'],
            features['hour_in_day'],
            features['minute_in_hour']
        ]]

        print("feature processed")
        # df = pd.DataFrame([d.model_dump() for d in data])
        # print("did dump")
        
        input_df = pd.DataFrame(input_data, columns=feature_names)
        print("converted to df")
        
        print(input_df)
        
        print('preparing to predict')
        prediction = model.predict(input_df)
        print('successfully predicted')

        print(prediction)
        prediction = int(prediction)
        predict_api_dict = {
            'Prediction': prediction,
            'df': df.to_dict(orient='records')
        }
        
        return predict_api_dict
    
    except Exception as e: 
        print("Error encountered")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("geolife_api:app", host="0.0.0.0", port=8000, reload=True)
    pass