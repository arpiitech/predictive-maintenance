FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY space/app.py /app/app.py

# Streamlit config
ENV HF_MODEL_REPO="arnavarpit/engine-predictive-maintenance-sklearn"
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
