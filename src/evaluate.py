import pandas as pd 
import pickle 
from sklearn.metrics import accuracy_score
import yaml 
import os 
import mlflow 
from urllib.parse import urlparse 
# --- DagsHub + MLflow setup ---
os.environ['MLFLOW_TRACKING_URI'] =  "https://dagshub.com/RogueNinja240/MLOPS_1.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "RogueNinja240"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "6d70d193d71c9f12397eb41b6d16fb88cd128927"

#params logging

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri = "https://dagshub.com/RogueNinja240/MLOPS_1.mlflow"

    #loading model
    model = pickle.load(open(model_path,'rb'))
    prediction = model.predict(X)
    accuracy = accuracy_score(y,prediction)

    #load metrics to MLFlow 
    mlflow.log_metric("accuracy",accuracy)

if __name__ == "__main__":
    evaluate(params["data"],params["model"])