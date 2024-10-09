import pandas as pd

if __name__ == "__main__":

    #Read csv
    data = pd.read_csv("../datasets/raw_data/predictive_maintenance.csv", sep=',') 

    #Data Preparation
    data = data.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)
    
    data.rename(
        columns = {
            'Air temperature [K]':'Air_temperature',
            'Process temperature [K]':'Process_temperature',
            'Rotational speed [rpm]':'Rotational_speed',
            'Torque [Nm]':'Torque',
            'Tool wear [min]':'Tool_wear',
        }, 
        inplace = True
    )

    data = pd.get_dummies(data)

    #Save processed dataset
    data.to_csv("../datasets/processed_data/processed_data.csv",sep=",")