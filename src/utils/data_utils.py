import pandas as pd

class DataPreprocessor():
    
    def __init__(self):
        
        self.features = [
                    'hour',
                    'minute',
                    'ambient_temperature',
                    'module_temperature',
                    'irradiation'
                ]

        self.targets = ['dc_power']

        return

    '''def extract_time_elements(self, data):

        data['hour'] = pd.to_datetime(data.date_time).dt.hour
        data['minute'] = pd.to_datetime(data.date_time).dt.minute

        return data
    
    def convert_datetime(self, data):

        data['hour'] = pd.to_datetime(data.date_time).dt.hour
        data['minute'] = pd.to_datetime(data.date_time).dt.minute

        return data'''
    
    #drops nulls
    def drop_nulls(self, data):
    
        data = data.dropna()

        return data
    
    #drops dc power (when irradiation is 0 dc power is 0)
    def drop_0_dcpower(self, data):
    
        data = data[data.dc_power>0]

        return data
    
    # precrocesses data
    def preprocess(self, data):
        
        data = self.drop_nulls(data)
        data = self.drop_0_dcpower(data)       
        
        df = data[self.targets + self.features].copy()
        
        return df

class InferenceDataPreprocessor(DataPreprocessor):
    
    def __init__(self):
        
        self.features = [
                    'hour',
                    'minute',
                    'ambient_temperature',
                    'module_temperature',
                    'irradiation'
                ]

        self.targets = ['dc_power']
        
        return