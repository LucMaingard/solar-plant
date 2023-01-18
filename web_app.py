from pydantic import BaseModel
from fastapi import FastAPI

import pandas as pd

from xgboost import XGBRegressor 

from src.training_job import train_model

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
@app.get('/{hour}/{minute}/{temp}/{mod_temp}/{irradiation}/{dc_power}/predict')
async def model_predict(hour:int, minute:int, temp:float, mod_temp:float, irradiation:float, dc_power:float):

    model_path = './models/model.json'
    dict_out = {}

    # if irradiation levels are above 0 (below = no power output)
    if irradiation>0:

        # load trained xgb model
        model = XGBRegressor()
        model.load_model(model_path)

        """Predict with input"""
        l1 = [hour, minute, temp, mod_temp, irradiation]

        mae = pd.read_csv('./data/processed_stats/best_model_scores.csv')['mae'].values

        x = pd.DataFrame(columns=['hour', 'minute', 'ambient_temperature', 'module_temperature', 'irradiation'],data=[l1])

        pred = model.predict(x)
        
        for count, value in enumerate(pred):
            dict_out['min_expected_dc_power'] = (float(value)-float(mae))
            dict_out['actual_dc_power'] = float(dc_power)

        if (float(value)-mae)>=(dc_power):
            dict_out['result']='low dc power output: needs maintenance'
        else:
            dict_out['result']='normal dc power output: working well'

    else:
        dict_out['min_expected_dc_power'] = float(0.0)
        dict_out['actual_dc_power'] = float(dc_power)
        dict_out['result']='normal dc power output: working well'

    

    return dict_out