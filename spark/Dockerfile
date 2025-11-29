FROM apache/spark:3.4.1

WORKDIR /app

USER root

# Install python dependencies
RUN pip install pyspark numpy

# Copy application
COPY spark_processor.py .

# We will use the spark-submit command in the job definition
