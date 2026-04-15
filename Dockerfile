FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader stopwords wordnet punkt

EXPOSE 8000

CMD ["python", "app.py"]