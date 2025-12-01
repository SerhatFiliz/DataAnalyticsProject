# Use Python 3.9 Slim image for a lightweight production environment.
FROM python:3.9-slim

# Set the working directory inside the container.
WORKDIR /app

# Install system-level Graphviz dependency required for architecture diagrams.
RUN apt-get update && apt-get install -y graphviz

# Copy the main application file into the container.
COPY app.py .

# Install necessary Python libraries including Streamlit and Graphviz wrapper.
RUN pip install streamlit pymongo pandas plotly scikit-learn graphviz

# Expose port 8501 to allow external access to the dashboard.
EXPOSE 8501

# Start the Streamlit server bound to all network interfaces.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]