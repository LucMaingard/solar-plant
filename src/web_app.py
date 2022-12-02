from pydantic import BaseModel
from fastapi import FastAPI

import pandas as pd

from xgboost import XGBRegressor 

from training_job import train_model

app = FastAPI()

# Input for data validation
class Input(BaseModel):
    hour: int
    minute: int
    ambient_temperature: float
    module_temperature: float
    irradiation: float
    dc_power: int 


#method to get 
@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": 'DC Power Generation Predictor',
        "version": 'v1'
    }

# method to check api health status
@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }

#method to call model training
@app.get('/train')
async def service_health():
    """Trigger model retraining"""
    training_result = train_model()

    return {
        "message": training_result
    }

# method to get prediction from most current model
@app.get('/{x1}/{x2}/{x3}/{x4}/{x5}/{x6}/predict')
async def model_predict(x1:int, x2:int, x3:float, x4:float, x5:float, x6:float):

    model_path = './app/solar/models/model.json'

    # load trained xgb model
    model = XGBRegressor()
    model.load_model(model_path)

    """Predict with input"""
    l1 = [x1, x2, x3, x4,x5]

    mae = pd.read_csv('/Users/lucmaingard/Dropbox/work/projects/solar/data/processed_stats/best_model_scores.csv')['mae'].values

    x = pd.DataFrame(columns=['hour', 'minute', 'ambient_temperature', 'module_temperature', 'irradiation'],data=[l1])

    pred = model.predict(x)
    dict_out = {}
    for count, value in enumerate(pred):
        dict_out['expected_dc_power'] = float(value)

    if (float(value)-mae)>=(x6):
        dict_out['result']='low dc power output: needs maintenance'
    else:
        dict_out['result']='normal dc power output: working well'

    return dict_out