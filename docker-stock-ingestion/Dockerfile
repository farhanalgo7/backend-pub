FROM python:3.8-slim-buster

# Add all files in the current directory to docker image
COPY . /docker-stock-ingestion

# add work dir
WORKDIR /docker-stock-ingestion

RUN apt-get update && pip3 install -r requirements.txt

ENTRYPOINT ["python3","stock_ingestion.py"]

