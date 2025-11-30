FROM python:3.9-slim

WORKDIR /app

COPY app.py .

# BURAYA scikit-learn EKLENDİ
RUN pip install streamlit pymongo pandas plotly scikit-learn

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]