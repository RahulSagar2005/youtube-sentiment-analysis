import matplotlib 
matplotlib.use('Agg')
from flask import Flask, request, jsonify, send_file 
from flask_cors import CORS 
import io 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud 
import mlflow 
import numpy as np 
import re
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import json

app = Flask(__name__)
CORS(app)

STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()


def preprocess_comment(comment):
    """Apply preprocessing steps to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = ' '.join(word for word in comment.split() if word not in STOP_WORDS)
        comment = ' '.join(lemmatizer.lemmatize(word) for word in comment.split())
        return comment
    except Exception as e:
        print(f"Error occurred while preprocessing comment: {e}")
        return comment


def load_model_and_vectorizer(vectorizer_path):
    """Load the model using run_id from experiment_info.json and vectorizer from local file."""
    try:
        mlflow.set_tracking_uri("http://ec2-98-93-179-250.compute-1.amazonaws.com:5000/")

        with open('experiment_info.json', 'r') as f:
            model_info = json.load(f)

        run_id = model_info['run_id']
        model_uri = f"runs:/{run_id}/lgbm_model"
        print(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully")

        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        print("Vectorizer loaded successfully")

        return model, vectorizer

    except FileNotFoundError:
        print("experiment_info.json not found — run the DVC pipeline first")
        raise
    except KeyError:
        print("run_id not found in experiment_info.json — check the file contents")
        raise
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        raise


model, vectorizer = load_model_and_vectorizer('tfidf_vectorizer.pkl')


@app.route('/')
def home():
    return "Welcome to the YouTube Sentiment Analysis API! Use the /predict endpoint to analyze comments."


# ✅ FIXED: handles both plain strings AND dicts {text, timestamp} from popup.js
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comments_input = data.get('comments')
    print("Received comments input type:", type(comments_input))

    if not comments_input:
        return jsonify({'error': 'No comments provided'}), 400

    try:
        comments_text = []
        timestamps = []
        for c in comments_input:
            if isinstance(c, dict):
                comments_text.append(c.get('text', ''))
                timestamps.append(c.get('timestamp', ''))
            else:
                comments_text.append(str(c))
                timestamps.append('')

        preprocessed_comments = [preprocess_comment(c) for c in comments_text]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        input_df = pd.DataFrame(dense_comments, columns=vectorizer.get_feature_names_out())
        predictions = model.predict(input_df)
        predictions = [int(p) for p in predictions]

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500

    # ✅ Returns comment + sentiment + timestamp so popup can use timestamp for trend graph
    response = [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp
        }
        for comment, sentiment, timestamp in zip(comments_text, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.get_json()
    comments_data = data.get('comments')
    if not comments_data:
        return jsonify({'error': 'No comments provided'}), 400
    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()

        input_df = pd.DataFrame(dense_comments, columns=vectorizer.get_feature_names_out())
        predictions = model.predict(input_df)
        predictions = [str(int(p)) for p in predictions]
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500

    response = [
        {"comment": comment, "timestamp": timestamp, "sentiment": sentiment}
        for comment, timestamp, sentiment in zip(comments, timestamps, predictions)
    ]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        if not sentiment_counts:
            return jsonify({'error': 'No sentiment counts provided'}), 400

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            sentiment_counts.get('1', 0),
            sentiment_counts.get('0', 0),
            sentiment_counts.get('-1', 0)
        ]
        if sum(sizes) == 0:
            return jsonify({'error': 'Sentiment counts cannot all be zero'}), 400

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='png', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f'Error generating chart: {e}')
        return jsonify({'error': f'Error generating chart: {e}'}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({'error': 'No comments provided'}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f'Error generating word cloud: {e}')
        return jsonify({'error': f'Error generating word cloud: {e}'}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiments')
        if not sentiment_data:
            return jsonify({'error': 'No sentiments provided'}), 400

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)