# Use the official Apache Spark base image for environment stability.
FROM apache/spark:3.4.1

# Set the working directory for the application code.
WORKDIR /app

# Temporarily switch to root user for dependency installation privileges.
USER root

# Install required Python dependencies for the Spark application logic.
RUN pip install pyspark numpy

# Copy the core Spark Structured Streaming and ML script.
COPY spark_processor.py .

# The final command (spark-submit) will be defined in the Kubernetes Job manifest.