import numpy as np 
import pandas as pd
import os  
import re 
import nltk 
import string 
from nltk.corpus import stopwords  # Fixed: 'nltkst.corpus' -> 'nltk.corpus'
from nltk.stem import WordNetLemmatizer 
import logging 

# Logging configuration 
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)  # Fixed: 'DEBUG' string -> logging.DEBUG constant

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Fixed: 'DEBUG' string -> logging.DEBUG constant

file_handler = logging.FileHandler('preprocessing_errors.log') 
file_handler.setLevel(logging.ERROR)  # Fixed: 'ERROR' string -> logging.ERROR constant

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter) 

logger.addHandler(console_handler)
logger.addHandler(file_handler) 

# Download required NLTK data 
nltk.download('stopwords') 
nltk.download('wordnet') 

# Initialize stopwords and lemmatizer once (not inside the function)
STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()


def preprocess_comment(comment):
    """Apply preprocessing steps to a single comment."""
    try:
        comment = comment.lower()
        comment = comment.strip() 
        comment = re.sub(r'\n', ' ', comment) 
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment) 
        comment = ' '.join([word for word in comment.split() if word not in STOP_WORDS])
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()]) 
        return comment 
    except AttributeError as e:
        logger.error(f"Error occurred while preprocessing comment: {e}")
        return comment


def normalize_text(df):
    """Apply preprocessing to the 'clean_comment' column of the DataFrame."""
    try:
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)  # Fixed: 'cleam_comment' -> 'clean_comment'
        logger.debug('Text normalization completed successfully.')
        return df
    except KeyError as e:
        logger.error(f"Column 'clean_comment' not found in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during text normalization: {e}")
        raise 


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed training and testing data to CSV files."""
    try:
        interim_data_path = os.path.join(data_path, 'interim') 
        logger.debug(f"Creating directory: {interim_data_path}")
        os.makedirs(interim_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(interim_data_path, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_data_path, 'test_processed.csv'), index=False)
        logger.debug(f"Preprocessed data saved successfully to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        raise 


def main():
    try:
        logger.debug('Starting data preprocessing...')
        train_data = pd.read_csv('./data/raw/train.csv')   # Fixed: '.data/raw/train.csv' -> './data/raw/train.csv'
        test_data = pd.read_csv('./data/raw/test.csv')     # Fixed: '.data/raw/test.csv'  -> './data/raw/test.csv'
        logger.debug('Data loaded successfully for preprocessing.') 

        # Preprocess the data 
        train_preprocessed_data = normalize_text(train_data)
        test_preprocessed_data = normalize_text(test_data)   

        # Save the processed data
        save_data(train_preprocessed_data, test_preprocessed_data, data_path='./data') 
    except Exception as e:
        logger.error('Failed to complete the data preprocessing: %s', e)
        raise  # Fixed: bare raise instead of just print, to preserve the exception


if __name__ == '__main__':
    main()