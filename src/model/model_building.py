import numpy as np 
import pandas as pd 
import os 
import pickle 
import yaml
import logging 
import lightgbm as lgb 
from sklearn.feature_extraction.text import TfidfVectorizer  # Fixed: wrong submodule path

logger = logging.getLogger('model_building') 
logger.setLevel(logging.DEBUG)  # Fixed: string 'DEBUG' -> logging.DEBUG constant

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Fixed: string 'DEBUG' -> logging.DEBUG constant

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.ERROR)  # Fixed: string 'ERROR' -> logging.ERROR constant

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter) 
file_handler.setFormatter(formatter) 

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info('Parameters loaded successfully from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise
    except Exception as e:
        logger.error('Unexpected error loading parameters from %s: %s', params_path, e)
        raise   


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except pd.errors.EmptyDataError as e:
        logger.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error loading data from %s: %s', file_path, e)
        raise 


def apply_tfidf_vectorization(train_data: pd.DataFrame, max_feature: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF vectorization to the training data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_feature, ngram_range=ngram_range)
        X_train = train_data['clean_comment'].values 
        y_train = train_data['category'].values 
        x_train_tfidf = vectorizer.fit_transform(X_train)
        logger.debug('TF-IDF vectorization applied successfully. Train shape: %s', x_train_tfidf.shape)

        vectorizer_path = os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as file:       # Fixed: variable was 'file' but pickle.dump used 'f'
            pickle.dump(vectorizer, file) 
        logger.debug('TF-IDF vectorizer saved successfully')
        return x_train_tfidf, y_train
    except KeyError as e:
        logger.error('Missing column for TF-IDF vectorization: %s', e)
        raise 


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:  # Fixed: lgb.LgbMBooster -> lgb.LGBMClassifier
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric='multi_logloss',
            is_unbalance=True,       # Fixed: 'ins_unbalance' -> 'is_unbalance'
            class_weight='balanced',
            reg_alpha=0.1,           # Fixed: 'reg_aplha' -> 'reg_alpha'
            reg_lambda=0.1,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('LightGBM model trained successfully')
        return best_model
    except Exception as e:
        logger.error('Error training LightGBM model: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved successfully to %s', file_path)
    except Exception as e:
        logger.error('Error saving model to %s: %s', file_path, e)
        raise 


def get_root_directory() -> str:
    """Get the root directory of the project.""" 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../')) 


def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        
        max_features = params['model_building']['max_features']       # Fixed: 'max_feature' -> 'max_features'
        ngram_range = tuple(params['model_building']['ngram_range'])  # Fixed: 'ngrams_range' -> 'ngram_range'
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        X_train_tfidf, y_train = apply_tfidf_vectorization(train_data, max_features, ngram_range)
        best_model = train_lightgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))  # Fixed: 'best_model.pkl' -> 'lgbm_model.pkl'
    except Exception as e:
        logger.error('Error in main function: %s', e)
        raise


if __name__ == '__main__':
    main()