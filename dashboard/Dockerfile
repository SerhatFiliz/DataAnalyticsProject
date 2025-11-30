FROM python:3.9-slim

WORKDIR /app

# 1. ÖNCE SİSTEM KÜTÜPHANESİNİ YÜKLE (Bu olmazsa grafik çizilmez hata verir)
RUN apt-get update && apt-get install -y graphviz

COPY app.py .

# 2. SONRA PYTHON KÜTÜPHANELERİNİ YÜKLE (graphviz'i buraya ekledik)
RUN pip install streamlit pymongo pandas plotly scikit-learn graphviz

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]