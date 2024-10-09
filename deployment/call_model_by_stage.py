import mlflow
import pandas as pd
import pickle

#Load categories
COLUMNS_PATH = "../src/ohe_categories.pkl"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)

#Load model by Stage
model_name = "My_model"
stage = "Production"

loaded_model = mlflow.pyfunc.load_model(
    model_uri = f"models:/{model_name}/{stage}"
)

# Input data
Type = "M"
Air_temperature	= 298.9
Process_temperature	= 309.1
Rotational_speed = 2861
Torque = 4.6
Tool_wear = 143

#Reformat data
data = {
    "Type":Type,
    "Air_temperature":Air_temperature,
    "Process_temperature":Process_temperature,
    "Rotational_speed":Rotational_speed,
    "Torque":Torque,
    "Tool_wear":Tool_wear
}

for key, value in data.items():
    data[key] = [value]

single_instance=pd.DataFrame.from_dict(data)

data_ohe = pd.get_dummies(single_instance).reindex(columns=ohe_tr).fillna(0)

#Prediction
response = loaded_model.predict(data_ohe)
print(response[0])