FROM python:3.8

COPY . /docker_tweets

WORKDIR /docker_tweets

RUN pip3 install --upgrade pip
RUN pip3 install setuptools==57.5.0
RUN apt-get update && pip3 install -r requirements.txt --no-cache-dir && python3 -m nltk.downloader punkt && python3 -m nltk.downloader stopwords
RUN pip install pycld2

ENTRYPOINT ["python","main.py"]