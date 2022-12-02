import numpy as np
import pandas as pd
import math
import os
import datetime

from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DCPowerPredictor():
    
    # initialise custom scorer
    def __init__(self):
        
        self.dc_power_predictor = XGBRegressor()

        return
    
    #fit data
    def fit(self, X, y):
            
        self.dc_power_predictor.fit(X, y.values.ravel())
        
        return

    # calculate mae, mse, rmse to evaluate model
    def evaluate(self, X, y):

        PATH = './solar/data/'

        preds = self.dc_power_predictor.predict(X)

        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        rmse = math.sqrt(mse)

        # Save as JSON file
        if not os.path.exists(PATH):
            
            # if the demo_folder directory is not present 
            # then create it.
            os.makedirs(PATH)

        return mae, mse, rmse
    
    # predict expected dc power
    def predict_expected_dc_power(self, X):
    
        y_pred = self.dc_power_predictor.predict(X)[0]
        
        return y_pred
    
    # saves model to current model location and versions model with timestamp for easy 
    def save_model(self):

        # model path
        MODEL_DIR_PATH = './solar/models/'

        # Save as JSON file
        if not os.path.exists(MODEL_DIR_PATH):
            
            # if the demo_folder directory is not present 
            # then create it.
            os.makedirs(MODEL_DIR_PATH)

        current_timestamp = str(datetime.datetime.now()).replace(" ","")
        VERSIONING_PATH = MODEL_DIR_PATH+current_timestamp+'/'

        if not os.path.exists(VERSIONING_PATH):
            
            # if the demo_folder directory is not present 
            # then create it.
            os.makedirs(VERSIONING_PATH)


        self.dc_power_predictor.save_model(MODEL_DIR_PATH+"model.json")
        self.dc_power_predictor.save_model(VERSIONING_PATH+"model.json")
        
        return 

    # load saved model
    def load_model(self, path):

        model = self.dc_power_predictor.load_mode(path+"model.json")
        
        return model
    
    
    
    