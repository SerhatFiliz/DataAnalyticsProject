# Use the official Python 3.9 Slim image for a minimal environment.
FROM python:3.9-slim

# Set the working directory within the container filesystem.
WORKDIR /app

# Copy the core consumer application logic.
COPY raw_consumer.py .

# Install necessary libraries: Kafka client for consumption and PyMongo for data storage.
RUN pip install kafka-python pymongo

# Define the container's entry point to start the consumer script.
CMD ["python", "raw_consumer.py"]