FROM python:3.9-slim

WORKDIR /app

COPY raw_consumer.py .

RUN pip install kafka-python pymongo

CMD ["python", "raw_consumer.py"]