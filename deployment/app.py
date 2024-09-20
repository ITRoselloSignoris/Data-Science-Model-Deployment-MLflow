import pandas as pd
import pickle
from fastapi import FastAPI

from fastapi.encoders import jsonable_encoder
import uvicorn
from pydantic import BaseModel


app = FastAPI()

COLUMNS_PATH = "../src/ohe_categories.pkl"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)


MODEL_PATH = "../src/model/rf.pkl"
model = pickle.load(open(MODEL_PATH, 'rb'))

class Answer(BaseModel):
    Age: int
    Class: int
    Wifi: int
    Booking: int
    Seat: int
    Checkin: int

@app.get("/")
async def root():
    return {"message":"Model Deployment Bootcamp"}

@app.post("/prediccion")
def predict_passenger_satisfaction(answer:Answer)
    answer_dict = jsonable_encoder(answer)
    
    for key, value in answer_dict.items():
        answer_dict[key] = [value]

    single_instance = pd.DataFrame.from_dict(answer_dict) 

    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns=ohe_tr).fillna(0)

    prediction = model.predict(single_instance_ohe)

    class_map = {1:"Yes", 0:"No"}

    score = prediction.map(class_map)

    response = {"Passenger Satisfaction: ":score}

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port="7260")