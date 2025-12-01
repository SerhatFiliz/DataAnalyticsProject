# Use Python 3.9 Slim image for a lightweight runtime environment.
FROM python:3.9-slim

# Set the working directory for the producer application.
WORKDIR /app

# Copy the core producer logic script into the container.
COPY producer.py .

# Install necessary Python libraries for Kafka communication and data handling.
RUN pip install kafka-python pandas

# Define the command to start the data streaming process.
CMD ["python", "producer.py"]