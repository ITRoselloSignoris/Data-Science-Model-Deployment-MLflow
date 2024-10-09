import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn
import pickle

if __name__ == "__main__":

    #Experiment name
    mlflow.set_experiment(experiment_name="proyecto_bootcamp_jueves")
    mlflow.set_tracking_uri('../../deployment/mlruns/')

    #Read csv
    data = pd.read_csv("../datasets/processed_data/processed_data.csv")

    #Separate between train and test
    x_data = data.drop('Target', axis=1)
    y_data = data['Target']

    TEST_SIZE = 0.3

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size= TEST_SIZE)

    #Train model
    N_ESTIMATORS = 5000
    RANDOM_STATE=19

    rf_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    rf_model.fit(x_train,y_train)

    #Metrics
    pred_ts = rf_model.predict(x_test)
    pred_tr = rf_model.predict(x_train)

    df_val_tr=pd.DataFrame({'y_train':y_train, 'pred_tr':pred_tr})
    df_val_ts=pd.DataFrame({'y_test':y_test, 'pred_ts':pred_ts})


    accuracy_test = accuracy_score(df_val_ts.y_test,df_val_ts.pred_ts, normalize=True)
    accuracy_train = accuracy_score(df_val_tr.y_train,df_val_tr.pred_tr, normalize=True)

    #Save One Hot Encoding Columns
    pickle.dump(x_data.columns,open("../ohe_categories.pkl","wb"),protocol=pickle.HIGHEST_PROTOCOL)

    #Save parameters to mlflow
    mlflow.log_param("Dataset size", data.shape)

    mlflow.log_param("Percentage of test size", TEST_SIZE)

    mlflow.log_param("Number of estimators", N_ESTIMATORS)

    mlflow.log_param("Random state", RANDOM_STATE)

    #Save metrics
    mlflow.log_metric("Train Accuracy ", accuracy_train)

    mlflow.log_metric("Test Accuracy ", accuracy_test)

    #Save model
    mlflow.sklearn.log_model(rf_model,"My_model")