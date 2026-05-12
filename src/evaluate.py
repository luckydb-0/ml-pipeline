import pandas as pd
import skops.io as sio
from sklearn.metrics import accuracy_score
import yaml
import mlflow
import os
from urllib.parse import urlparse

# Set env variables for DagsHub and MLFlow
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/luckydb-0/ml-pipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'luckydb-0'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '...'

params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)

    X = data.drop(columns = ['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # Load model from the disk
    unknown_types = sio.get_untrusted_types(file=model_path)
    if unknown_types:
        print("Skops toy example. Untrusted types found:")
        for t in unknown_types:
            print(f" - {t}")

    model = sio.load(model_path, trusted = unknown_types)

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log metrics to MLFlow
    mlflow.log_metric('accuracy', accuracy)
    print(f'Model accuracy: {accuracy}')


if __name__ == '__main__':
    evaluate(params['data'], params['model'])