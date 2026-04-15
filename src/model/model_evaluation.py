import numpy as np
import pandas as pd 
import pickle 
import logging 
import yaml 
import mlflow 
import mlflow.sklearn 
from sklearn.metrics import classification_report, confusion_matrix  # Fixed: trailing comma removed
from sklearn.feature_extraction.text import TfidfVectorizer
import os  
import matplotlib.pyplot as plt
import seaborn as sns
import json 
from mlflow.models import infer_signature 

logger = logging.getLogger('model_evaluation') 
logger.setLevel(logging.DEBUG)  # Fixed: string 'DEBUG' -> logging.DEBUG constant

console_handler = logging.StreamHandler() 
console_handler.setLevel(logging.DEBUG)  # Fixed: string 'DEBUG' -> logging.DEBUG constant

file_handler = logging.FileHandler('model_evaluation_error.log') 
file_handler.setLevel(logging.ERROR)  # Fixed: string 'ERROR' -> logging.ERROR constant

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter) 

logger.addHandler(console_handler) 
logger.addHandler(file_handler) 


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading from %s: %s', file_path, e) 
        raise


def load_model(model_path: str):
    """Load a model from a pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded successfully from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise 


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load a vectorizer from a pickle file."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('Vectorizer loaded successfully from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise 


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded successfully from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise 


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):  # Fixed: 'evalaute_model' -> 'evaluate_model'
    """Evaluate the model and return evaluation metrics."""
    try:
        y_pred = model.predict(X_test)  # Fixed: 'X_text' -> 'X_test'
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug('Model evaluation completed successfully')
        return report, cm
    except Exception as e:
        logger.error('Error evaluating model: %s', e)
        raise 


def log_confusion_matrix(cm, dataset_name):
    """Log the confusion matrix as an artifact."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        os.remove(cm_file_path) 
        logger.debug('Confusion matrix logged successfully for %s', dataset_name) 
        plt.close()
    except Exception as e:
        logger.error('Error logging confusion matrix for %s: %s', dataset_name, e)
        raise 


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save model information to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file)
        logger.debug('Model information saved successfully to %s', file_path)
    except Exception as e:
        logger.error('Error saving model information to %s: %s', file_path, e)
        raise 


def main():
    mlflow.set_tracking_uri("http://ec2-98-93-179-250.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment('dvc-pipeline-runs') 
    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))  # Fixed: nested os.path.dirname call was wrong
            params = load_params(os.path.join(root_dir, 'params.yaml')) 

            for key, value in params.items():
                mlflow.log_params({key: value})  # Fixed: 'key:value' dict literal syntax error -> '{key: value}'

            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values 

            input_example = pd.DataFrame(X_test_tfidf[:5].toarray(), columns=vectorizer.get_feature_names_out())  # Fixed: sparse matrix needs .toarray()
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5])) 
            mlflow.sklearn.log_model(model, 'lgbm_model', signature=signature, input_example=input_example) 

            artifact_url = mlflow.get_artifact_uri()
            model_path = f"{artifact_url}/lgbm_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json') 
            mlflow.log_artifact('experiment_info.json')  # Fixed: no need for os.path.join, file is saved locally

            report, cm = evaluate_model(model, X_test_tfidf, y_test)  # Fixed: 'evalaute_model' -> 'evaluate_model'
            for label, metrics in report.items(): 
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    }) 
            log_confusion_matrix(cm, 'Test Data') 

            mlflow.set_tag('model_type', 'LightGBM')
            mlflow.set_tag('test', 'Sentiment Analysis')
            mlflow.set_tag('dataset', 'Youtube Comments')

        except Exception as e:
            logger.error('Error in main execution: %s', e)
            raise  # Fixed: bare raise instead of just print


if __name__ == '__main__':
    main()