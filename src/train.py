import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import skops.io as sio
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
import mlflow

from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

# Set env variables for DagsHub and MLFlow
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/luckydb-0/ml-pipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'luckydb-0'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '...'

# Load parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['train']

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)

    return grid_search

# Last two params currently not used, present just as an example
def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)

    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # Start MLFlow run
    with mlflow.start_run():
        # Split dataset into training and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = random_state)
        signature = infer_signature(X_train, y_train)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')

        # Log additional metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param('best_n_estimators', grid_search.best_params_['n_estimators'])
        mlflow.log_param('best_max_depth', grid_search.best_params_['max_depth'])
        mlflow.log_param('best_sample_split', grid_search.best_params_['min_samples_split'])
        mlflow.log_param('best_samples_leaf', grid_search.best_params_['min_samples_leaf'])

        # Log the confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), 'confusion_matrix.txt')
        mlflow.log_text(str(cr), 'classification_report.txt')

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file': # Call to HTTP/S
            mlflow.sklearn.log_model(best_model, name = 'model', registered_model_name = 'Best model')
        else:
            mlflow.sklearn.log_model(best_model, name = 'model', signature = signature)

        # Save the model locally
        os.makedirs(os.path.dirname(model_path), exist_ok = True)
        sio.dump(best_model, model_path) # models/model.skops
        print(f'Model saved to {model_path}')


if __name__ == '__main__':
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])
