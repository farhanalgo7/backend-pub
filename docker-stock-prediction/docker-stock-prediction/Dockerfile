FROM python:3.8-slim-buster

RUN python3 -m pip install -U pip && pip install numpy 
RUN apt-get update && apt-get -y install gcc
RUN apt-get update  && apt-get install -y wget && apt-get install make
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

COPY . /usr/app/

# add work dir
WORKDIR /usr/app

RUN pip3 install -r requirements.txt


#ENTRYPOINT ["python3" , "stock_prediction_2.py"]










