import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import pickle
import os

app = Flask(__name__)
CORS(app)

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
lemmatizer = WordNetLemmatizer()

# ✅ Load API key from environment variable (never hardcode it)
YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY environment variable is not set. Please set it before starting the server.")


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


def load_model_and_vectorizer():
    """Load model and vectorizer directly from pkl files."""
    try:
        with open('/app/lgbm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")

        with open('/app/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("Vectorizer loaded successfully")

        return model, vectorizer
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        raise


# Load model and vectorizer at startup
model, vectorizer = load_model_and_vectorizer()


@app.route('/')
def home():
    return "Welcome to the YouTube Sentiment Analysis API! Use the /predict endpoint to analyze comments."


@app.route('/comments')
def get_comments():
    """Proxy endpoint to fetch YouTube comments using server-side API key."""
    video_id = request.args.get("video_id")
    page_token = request.args.get("pageToken", "")

    if not video_id:
        return jsonify({"error": "No video_id provided"}), 400

    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": YOUTUBE_API_KEY,   # ✅ Key is never sent to the client
        "maxResults": 100,
    }

    # ✅ Only include pageToken if it's not empty
    if page_token:
        params["pageToken"] = page_token

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": f"YouTube API error: {response.status_code}", "details": response.text}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint for Docker containers."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for comments."""
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

    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments_text, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Predict sentiment for comments with timestamps."""
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
    """Generate a pie chart showing sentiment distribution."""
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
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
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
    """Generate a word cloud from comments."""
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({'error': 'No comments provided'}), 400

        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            colormap='viridis', stopwords=set(stopwords.words('english')),
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
    """Generate a trend graph showing sentiment over time."""
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
            plt.plot(monthly_percentages.index, monthly_percentages[sentiment_value],
                     marker='o', linestyle='-', label=sentiment_labels[sentiment_value],
                     color=colors[sentiment_value])

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
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)