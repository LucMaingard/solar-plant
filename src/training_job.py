import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor 

from utils.data_utils import *
from utils.model_utils import *

# cleans data 
def clean_data(df_p1, df_w1):

    df_p1=lowercase_cols(df_p1)
    df_w1=lowercase_cols(df_w1)

    df_p1, df_w1 = extract_time_elements(df_p1, df_w1)
    df = join_dfs(df_p1, df_w1)

    return df

#drops dc power (when irradiation is 0 dc power is 0)
def lowercase_cols(df):

    #lowercase all column names
    df.columns=df.columns.str.lower()

    return df

# gets time elements 
def extract_time_elements(df_p1, df_w1):

    df_p1['time'] = pd.to_datetime(df_p1.date_time).dt.time
    df_p1['date'] = pd.to_datetime(df_p1.date_time).dt.date
    df_p1['hour'] = pd.to_datetime(df_p1.date_time).dt.hour
    df_p1['minute'] = pd.to_datetime(df_p1.date_time).dt.minute

    df_w1['time'] = pd.to_datetime(df_w1.date_time).dt.time
    df_w1['date'] = pd.to_datetime(df_w1.date_time).dt.date

    return df_p1, df_w1

# joins dfs together 
def join_dfs(df_p1, df_w1):

    df = pd.merge(df_p1.drop(columns = ['plant_id']), df_w1.drop(columns = ['plant_id', 'source_key']), on=('date', 'time'), how='left')

    return df

# holds all preprocessing steps
def preprocess_train_data(data):
        
    preprocessor = DataPreprocessor()
    data_preprocessed = preprocessor.preprocess(data)
    
    X = data_preprocessed.iloc[:,1:].copy()

    y = data_preprocessed['dc_power'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

# predict function
def predict_fn(input_object, model):
    
    preprocessor = InferenceDataPreprocessor()
    data_preprocessed = preprocessor.preprocess(input_object)
    
    prediction = model.predict_expected_dc_power(data_preprocessed)
    
    return prediction

def get_model_response(input_object, model, mae):

    # converts json into df
    df = pd.json_normalize(input_object)
    X = df.iloc[:,:-1]
    true_dc_power = df.iloc[:,-2:]
    prediction = model.predict(X)[0]
    
    # if the true value is below 2 stdeviations of the expected value, flag as 
    if float(true_dc_power-prediction) >= float(mae*2):
        label = "Maintenance required"
    else:
        label = "Panel is performing well"

    return {
        'label': str(label),
        'expected_dc_power_gen': prediction,
        'actual_dc_power_gen': true_dc_power
    }

# reads the rmse of the best performing model 
# used to assess whether or not newly trained model should be deployed (only if rmse is higher, can be changed to include all metrics)  
def get_rmse():

    rmse = pd.read_csv('/Users/lucmaingard/Dropbox/work/projects/solar/data/processed_stats/best_model_scores.csv')['rmse'].values

    return rmse

# function to train and evaluate not model (model is deployed if its rmse is better than the previous model)
def train_model():

    p1_url = '/Users/lucmaingard/Dropbox/work/projects/solar/data/raw/Plant_1_Generation_Data.csv'
    w1_url = '/Users/lucmaingard/Dropbox/work/projects/solar/data/raw/Plant_1_Weather_Sensor_Data.csv'

    df_p1 = pd.read_csv(p1_url)
    df_w1 = pd.read_csv(w1_url)
    print("fetched data")

    #gets data
    data = clean_data(df_p1, df_w1)
    print("cleaned data")

    #processes data
    X_train, X_test, y_train, y_test = preprocess_train_data(data)
    print("processed data")

    scorer = DCPowerPredictor()
    scorer.fit(X_train, y_train)
    print("fit model data")

    #get model metrics
    mae, mse, rmse = scorer.evaluate(X_test, y_test)
    prev_best_rmse = get_rmse()

    print(f"\nnew model rmse: {rmse} vs old model rmse: {prev_best_rmse}")

    if rmse<=prev_best_rmse:

        scorer.save_model()
        df = pd.DataFrame(columns=['mae', 'mse', 'rmse'], data=[[mae, mse, rmse]])
        df.to_csv('/Users/lucmaingard/Dropbox/work/projects/solar/data/processed_stats/best_model_scores.csv')

        print('The new model outperformed the old one and was successfully deployed :)')
        result = 'The new model outperformed the old one and was successfully deployed :)'
    else:
        print("The new model was not an improvement on the previous model. It was not deployed :(")
        result = "The new model was not an improvement on the previous model. It was not deployed :("

    return result

if __name__ == "__main__":
    print('its working')

    train_model()
