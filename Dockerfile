FROM python:3.8.5-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "app.py"]