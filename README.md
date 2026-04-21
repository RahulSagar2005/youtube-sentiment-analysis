# YouTube Sentiment Analysis

A machine learning-powered sentiment analysis system for YouTube comments, deployed on AWS with a complete CI/CD pipeline. This project analyzes YouTube comment sentiment (Positive, Neutral, Negative) using a LightGBM classifier with TF-IDF vectorization.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Deployment on AWS](#deployment-on-aws)
- [CI/CD Pipeline](#cicd-pipeline)
- [Usage](#usage)
- [Configuration](#configuration)
- [Monitoring & Experiment Tracking](#monitoring--experiment-tracking)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Real-time Sentiment Analysis**: Analyze YouTube comment sentiment in real-time via REST API
- **Three-Class Classification**: Positive (1), Neutral (0), Negative (-1)
- **Data Pipeline**: Automated ETL pipeline using DVC for reproducible ML workflows
- **MLflow Integration**: Experiment tracking and model registry
- **Dockerized Deployment**: Containerized application for consistent deployments
- **AWS Native**: Deployed on EC2 with ECR for container registry
- **CI/CD Automation**: GitHub Actions for continuous integration and deployment
- **Visualization APIs**: Generate pie charts, word clouds, and sentiment trend graphs
- **Chrome Extension Ready**: Backend API designed to support browser extension integration

---

## Tech Stack

### Core Technologies

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11 |
| **ML Framework** | LightGBM 4.5.0 |
| **Vectorization** | TF-IDF (scikit-learn) |
| **ML Operations** | MLflow 2.11.3 |
| **Data Pipeline** | DVC (Data Version Control) 3.53.0 |
| **Web Framework** | Flask 3.0.3 |
| **Containerization** | Docker |
| **Cloud Provider** | AWS (EC2, ECR) |

### Key Libraries

- **Data Processing**: pandas, numpy, nltk
- **Visualization**: matplotlib, seaborn, wordcloud
- **ML/Stats**: scikit-learn 1.5.2, joblib
- **Cloud**: boto3
- **API**: Flask-CORS

---

## Project Structure

```
youtube-sentiment-analysis/
├── .github/workflows/
│   └── cicd.yaml              # GitHub Actions CI/CD pipeline
├── src/
│   ├── data/
│   │   ├── data_ingestion.py      # Fetches and splits raw data
│   │   └── data_preprocessing.py  # Text cleaning and normalization
│   └── model/
│       ├── model_building.py      # TF-IDF + LightGBM training
│       ├── model_evaluation.py    # Model performance evaluation
│       └── register_model.py      # Registers model in MLflow
├── flask_api/
│   └── main.py                # Flask API server (alternative entry point)
├── data/
│   ├── raw/                   # Raw train/test CSV files
│   └── interim/               # Processed data
├── notebooks/                 # Jupyter notebooks for exploration
├── app.py                     # Main Flask API application
├── Dockerfile                 # Container definition
├── dvc.yaml                   # DVC pipeline configuration
├── dvc.lock                   # DVC pipeline lock file
├── params.yaml                # Hyperparameters configuration
├── requirements.txt           # Python dependencies
├── lgbm_model.pkl             # Trained LightGBM model
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
├── experiment_info.json       # MLflow run metadata
└── README.md                  # This file
```

---

## Data Pipeline

The ML pipeline is orchestrated using **DVC (Data Version Control)** with the following stages:

### Stage 1: Data Ingestion
- **Source**: Reddit Sentiment Analysis dataset (GitHub)
- **Action**: Downloads CSV, splits into train/test (80/20)
- **Output**: `data/raw/train.csv`, `data/raw/test.csv`

### Stage 2: Data Preprocessing
- **Cleaning**: Lowercase, remove special characters, strip whitespace
- **Normalization**: Remove stopwords (preserving negations), lemmatization
- **Output**: `data/interim/train_processed.csv`, `data/interim/test_processed.csv`

### Stage 3: Model Building
- **Vectorization**: TF-IDF with configurable max features and n-gram range
- **Training**: LightGBM multiclass classifier
- **Output**: `lgbm_model.pkl`, `tfidf_vectorizer.pkl`

### Stage 4: Model Evaluation
- **Metrics**: Accuracy, precision, recall, F1-score
- **Tracking**: Logs metrics to MLflow
- **Output**: `experiment_info.json` (contains MLflow run_id)

### Stage 5: Model Registration
- **Action**: Registers best model in MLflow model registry
- **Output**: Model available for inference

---

## Model Architecture

### LightGBM Classifier Configuration

```yaml
objective: multiclass
num_class: 3                    # Positive, Neutral, Negative
metric: multi_logloss
is_unbalance: true              # Handle class imbalance
class_weight: balanced
reg_alpha: 0.1                  # L1 regularization
reg_lambda: 0.1                 # L2 regularization
learning_rate: 0.09
max_depth: 20
n_estimators: 367
```

### TF-IDF Vectorization

```yaml
max_features: 1000
ngram_range: [1, 3]             # Unigrams, bigrams, trigrams
```

---

## API Endpoints

The Flask API provides the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check for Docker containers |
| `/predict` | POST | Predict sentiment for comments |
| `/predict_with_timestamps` | POST | Predict with timestamp metadata |
| `/generate_chart` | POST | Generate sentiment distribution pie chart |
| `/generate_wordcloud` | POST | Generate word cloud from comments |
| `/generate_trend_graph` | POST | Generate sentiment trend over time |

### Request/Response Examples

**POST /predict**
```json
{
  "comments": [
    {"text": "Great video!", "timestamp": "2024-01-01T10:00:00Z"},
    {"text": "Terrible content", "timestamp": "2024-01-01T10:05:00Z"}
  ]
}
```

**Response**
```json
[
  {"comment": "Great video!", "sentiment": 1, "timestamp": "2024-01-01T10:00:00Z"},
  {"comment": "Terrible content", "sentiment": -1, "timestamp": "2024-01-01T10:05:00Z"}
]
```

---

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/youtube-sentiment-analysis.git
cd youtube-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader stopwords wordnet punkt

# Run the DVC pipeline
dvc repro

# Start the API server
python app.py
```

### Docker

```bash
# Build the image
docker build -t youtube-sentiment-analysis .

# Run the container
docker run -p 8000:8000 youtube-sentiment-analysis
```

---

## Deployment on AWS

The application is deployed on AWS using the following architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                            │
│                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌────────────┐ │
│   │   GitHub    │─────▶│     ECR     │─────▶│    EC2     │ │
│   │  Actions    │      │  Container  │      │  Instance  │ │
│   │  (CI/CD)    │      │   Registry  │      │  (Flask)   │ │
│   └─────────────┘      └─────────────┘      └────────────┘ │
│                                                 │           │
│                                                 ▼           │
│                                          ┌────────────┐     │
│                                          │   MLflow   │     │
│                                          │  Tracking  │     │
│                                          └────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Infrastructure Components

1. **Amazon ECR (Elastic Container Registry)**
   - Stores Docker container images
   - Versioned image tags for rollback capability

2. **Amazon EC2**
   - Hosts the Flask API container
   - Instance type: t2.medium or higher recommended
   - Security group: Allow inbound traffic on port 8000

3. **MLflow Tracking Server**
   - Remote tracking URI for experiment management
   - Model registry for versioned models

### Deployment Steps

1. **Configure AWS Credentials**
   ```bash
   aws configure
   ```

2. **Create ECR Repository**
   ```bash
   aws ecr create-repository --repository-name youtube-sentiment-analysis
   ```

3. **Set GitHub Secrets**
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `ECR_REPOSITORY_NAME`
   - `AWS_ECR_LOGIN_URI`
   - `MLFLOW_TRACKING_URI`

4. **Push to Main Branch**
   - CI/CD pipeline automatically builds and deploys

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/cicd.yaml`) consists of three jobs:

### Job 1: Continuous Integration
- Checkout code
- Run linting
- Execute unit tests

### Job 2: Build and Push to ECR
- Configure AWS credentials
- Login to Amazon ECR
- Build Docker image
- Tag and push to ECR

### Job 3: Continuous Deployment
- Pull latest image from ECR on EC2
- Stop existing container
- Run new container with environment variables
- Clean up old images

### Pipeline Flow

```
push to main ──▶ lint/test ──▶ build image ──▶ push to ecr ──▶ deploy to ec2
```

---

## Usage

### Running the Pipeline

```bash
# Run full DVC pipeline
dvc repro

# Run specific stage
dvc repro data_ingestion
dvc repro data_preprocessing
dvc repro model_building
dvc repro model_evaluation
dvc repro model_registration
```

### API Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Sentiment prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["I love this!", "This is awful", "Its okay"]}'

# Generate word cloud
curl -X POST http://localhost:8000/generate_wordcloud \
  -H "Content-Type: application/json" \
  -d '{"comments": ["amazing", "fantastic", "terrible", "boring"]}' \
  --output wordcloud.png
```

---

## Configuration

### params.yaml

```yaml
data_ingestion:
  test_size: 0.20

model_building:
  ngram_range: [1, 3]
  max_features: 1000
  learning_rate: 0.09
  max_depth: 20
  n_estimators: 367
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | API server port | 8000 |
| `MLFLOW_TRACKING_URI` | MLflow server URL | http://localhost:5000 |
| `AWS_ACCESS_KEY_ID` | AWS access key | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - |
| `AWS_REGION` | AWS region | us-east-1 |

---

## Monitoring & Experiment Tracking

### MLflow Integration

- **Tracking**: All experiments logged with parameters, metrics, and artifacts
- **Model Registry**: Versioned model storage with stage transitions
- **Access**: Navigate to MLflow UI at configured tracking URI

### Logs

Application logs are written to:
- `errors.log` - Data ingestion errors
- `preprocessing_errors.log` - Preprocessing errors
- `model_building.log` - Training logs
- `model_evaluation_error.log` - Evaluation errors
- `model_registration_error.log` - Registration errors

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [Reddit Sentiment Analysis](https://github.com/Himanshu-1703/reddit-sentiment-analysis)
- ML Framework: [LightGBM](https://lightgbm.readthedocs.io/)
- Experiment Tracking: [MLflow](https://mlflow.org/)
- Data Version Control: [DVC](https://dvc.org/)
