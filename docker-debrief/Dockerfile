FROM python:3.8-slim-buster

# Add all files in the current directory to docker image
COPY . /usr/app/

# add work dir
WORKDIR /usr/app

RUN python3 -m pip install -U pip && apt-get update && apt-get install -y git && pip3 install -r requirements.txt

ENTRYPOINT ["python3","main.py"]