
FROM python:3.8-slim-buster AS build
WORKDIR /AKS-Microservices

RUN mkdir model
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN wget "https://fabric-ml-models.s3.ap-south-1.amazonaws.com/config.json" -P /AKS-Microservices/event_classification_model
RUN wget "https://fabric-ml-models.s3.ap-south-1.amazonaws.com/model_90.joblib" -P /AKS-Microservices/event_classification_model
RUN wget "https://fabric-ml-models.s3.ap-south-1.amazonaws.com/special_tokens_map.json" -P /AKS-Microservices/event_classification_model
RUN wget "https://fabric-ml-models.s3.ap-south-1.amazonaws.com/tf_model.h5" -P /AKS-Microservices/event_classification_model
RUN wget "https://fabric-ml-models.s3.ap-south-1.amazonaws.com/tokenizer_config.json" -P /AKS-Microservices/event_classification_model
RUN wget "https://fabric-ml-models.s3.ap-south-1.amazonaws.com/vocab.txt" -P /AKS-Microservices/event_classification_model

COPY event_classification_model new_main_v4.py requirements.txt ./

RUN apt-get update && pip3 install -r requirements.txt
RUN python3 -m pip install spacy download en_core_web_sm

RUN ls -a 

ENTRYPOINT ["python", "new_main_v4.py" ]
