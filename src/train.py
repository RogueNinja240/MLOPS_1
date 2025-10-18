import os
import yaml
import pickle
import mlflow
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- DagsHub + MLflow setup ---
os.environ['MLFLOW_TRACKING_URI'] =  "https://dagshub.com/RogueNinja240/MLOPS_1.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "RogueNinja240"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "6d70d193d71c9f12397eb41b6d16fb88cd128927"

# Function for hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load params from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    target_col = params.get("target", "Outcome")  # Default fallback
    X = data.drop(columns=[target_col])
    y = data[target_col]

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("RandomForest_Tuning")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
        signature = infer_signature(X_train, y_train)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # Hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        # Evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"✅ Accuracy: {accuracy:.4f}")
        print(f"✅ Best Parameters: {grid_search.best_params_}")

        # Log parameters & metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Confusion matrix & classification report
        conf_mx = confusion_matrix(y_test, y_pred)
        class_rpt = classification_report(y_test, y_pred)
        mlflow.log_text(str(conf_mx), "confusion_matrix.txt")
        mlflow.log_text(str(class_rpt), "classification_report.txt")

        # Save model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, "wb"))
        print(f"💾 Model saved locally at {model_path}")

        # Log local model as artifact (instead of log_model)
        tracking_uri = mlflow.get_tracking_uri().lower()
        if "dagshub" in tracking_uri:
            print("🌐 DagsHub detected — logging model as artifact.")
            mlflow.log_artifact(model_path, artifact_path="saved_model")
        else:
            print("📦 Local MLflow detected — logging full model with signature.")
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Also log your params.yaml file for reproducibility
        mlflow.log_artifact("params.yaml")

if __name__ == "__main__":
    train(
        params['data'],
        params['model'],
        params['random_state'],
        params['n_estimators'],
        params['max_depth']
    )
