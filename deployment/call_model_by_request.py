import pandas as pd
import pickle
import requests
import json

# RECORDAR LEVANTAR EL SERVER, example port: 1234
# mlflow models serve --model-uri models:/My_model/Production -p 1234 --no-conda

#Local Host Url
url = "http://127.0.0.1:1234/invocations"

#Load categories
COLUMNS_PATH = "../src/ohe_categories.pkl"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)

#Input data
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
data_ohe = data_ohe.values.tolist()

request_data = {
		"dataframe_records":data_ohe
}

#POST
response = requests.post(url,json=request_data)
print (response.json())