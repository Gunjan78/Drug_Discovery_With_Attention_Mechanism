FROM python:3.10.14-bookworm

RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY drug_molecule_gen /app/drug_molecule_gen

COPY tests /app/tests

COPY main.py /app

WORKDIR /app

RUN mkdir /app/app_data

RUN chmod -R 777 /app/app_data

ENV PYTHONPATH=${PYTHONPATH}:/app/drug_molecule_gen

RUN pip install --no-cache-dir -r /app/drug_molecule_gen/requirements.txt

EXPOSE 8000

RUN chmod +x /app/drug_molecule_gen/train_pipeline.py

RUN chmod +x /app/drug_molecule_gen/predict.py

RUN chmod +x /app/main.py

#ENTRYPOINT ["python3"]

CMD pip install -e .