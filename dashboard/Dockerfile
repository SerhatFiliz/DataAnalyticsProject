FROM python:3.9-slim

WORKDIR /app

COPY app.py .

RUN pip install streamlit pymongo pandas plotly

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]