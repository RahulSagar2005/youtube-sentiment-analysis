import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import yaml
import logging 

# Logging configuration 
logger = logging.getLogger('data_ingestion') 
logger.setLevel(logging.DEBUG) 

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) 
file_handler = logging.FileHandler('errors.log') 
file_handler.setLevel(logging.ERROR) 

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
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url) 
        logger.debug('Data loaded successfully from %s', data_url)
        return df
    except pd.errors.EmptyDataError:
        logger.error('No data: %s is empty', data_url)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise 


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by dropping missing values, duplicates, and empty strings."""
    try:
        df = df.dropna()
        df = df.drop_duplicates()
        df = df[df['clean_comment'].str.strip() != '']
        logger.debug('Data preprocessed successfully, missing values dropped')
        return df
    except KeyError as e:
        logger.error('Missing expected column: %s', e)
        raise
    except Exception as e:
        logger.error('Error during data preprocessing: %s', e)
        raise 


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train/test data to CSV files."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)  # Fixed: was using os.path.dirname on a directory path
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug('Preprocessed data saved successfully to %s', raw_data_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise 


def main():
    try:
        # Load parameters from the params.yaml file in the root directory
        params = load_params(
            params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml')
        ) 
        test_size = params['data_ingestion']['test_size'] 

        # Load data from the CSV file 
        df = load_data(
            data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        ) 

        # Preprocess the data
        final_df = preprocess_data(df) 

        # Split the data into training and testing sets 
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42) 

        # Save data
        save_data(
            train_data, test_data,
            data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
        )
    except Exception as e:
        logger.error('Error in main function: %s', e)
        raise   


if __name__ == '__main__':
    main()