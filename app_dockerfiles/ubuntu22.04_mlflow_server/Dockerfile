FROM python:3.10.14-bookworm

# Upgrade pip
RUN pip install --upgrade pip

# Install MLflow
RUN pip install mlflow==2.12.2

# Install MySQL connector for Python
RUN pip install mysql-connector-python mysqlclient

# Expose the default MLflow port
EXPOSE 5000

# Create directory for MLflow artifacts
RUN mkdir -p /mlruns && chmod 777 /mlruns

# Start the MLflow server
CMD mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri
