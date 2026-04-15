FROM python:3.10-slim          # Fixed: 3.8.5-slim-buster -> 3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000