


''' 
Fast API for Name Entity Recognizer (NER) and Sentiment Analysis Microservice  
 
1. Recognizes entitie's in article's (NER)
2. Sentiment prediction for related entities [ORG].
3. API returns NER Analysis and Sentiments Analysis on [POST] request

Prerequisites
  !pip install pandas
  !pip install spacy==3.1.1 
  !pip install fastapi
  !pip install uvicorn[standard]
  !pip install transformers==4.6.1
  !pip install torch==1.9.0
  !python -m spacy download en_core_web_md 

Author: Shreyas R Chim <schim@alagoanalytics.com>

Created: 3rd Dec 2021

'''

from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import numpy as np
from tqdm import tqdm
import json 
from typing import List
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM) # sentiment and summarization 

import torch.nn.functional as F

from finbert_embedding.embedding import FinbertEmbedding        # for sentence embedding   
from joblib import dump, load     # for loading trained event detection model
import nltk   # for sentence tokenization
nltk.download('punkt')
 
NER_MODEL_NAME = 'en_core_web_md'
SENTIMENT_MODEL_NAME = 'ProsusAI/finbert'
SUMMARIZATION_MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
EVENT_DETECTION_MODEL_PATH = 'model_90.joblib'

### Part: Load the models
# Model 1: Load the models for NER analysis
print(f"Loading the {NER_MODEL_NAME} model...")
# Load Spacy transformer model
sp = spacy.load("en_core_web_md", exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
sp.add_pipe('sentencizer')     # add a new pipeline. It allow custom sentence boundary detection logic that doesn’t require the dependency parse.


# Model 2: Load the model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME,local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

# Model 3: Load the model for summarization
print(f"Loading the {SUMMARIZATION_MODEL_NAME} model...")
summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)

# Model 4: Load the model for event/unknown classification
print("Loading the event detection model...")
event_tokenizer =  nltk.sent_tokenize
event_model = load(EVENT_DETECTION_MODEL_PATH)
finbert = FinbertEmbedding()

app = FastAPI( title="FABRIC Microservice | NER_SENTIMENT_SUMMARIZATION Model APIs",
    description="Exposes NLP models as API endpoints using FastApi",
    version="0.0.1",
    contact={
        "name": "Shreyas Chim",
        "email": "schim@algoanalytics.com",
    },)




def get_sentiment_class(news_sent_4,news_sent_5,news_sent_6):
    if (news_sent_4 > news_sent_5) and (news_sent_4 > news_sent_6):
      sentiment_class = 'negative'
    elif (news_sent_5> news_sent_4) and (news_sent_5 > news_sent_6):
      sentiment_class = 'positive'
    elif (news_sent_6> news_sent_4) and (news_sent_6 > news_sent_5):
      sentiment_class = 'neutral' 
    return sentiment_class


def analyze_sentiment(doc):
    """ Function to perform Sentiment analysis using ProsusAI/finbert tokenizer and pretrained NLP model.
    
    The function will take a doc/text as input and first apply tokenizer and then calculate outputs by using model function on it. 
    We will perform softmax on outputs and return numpy array of pt_predictions.

    Parameters
      -------------
            1. doc : a single string

      Returns
      ----------
        pt_predictions.detach().cpu().numpy(): numpy array of pt_predictions 

    """
    pt_batch = tokenizer(doc,padding=True,truncation=True,max_length=512,return_tensors="pt")
    outputs = model(**pt_batch)
    pt_predictions = F.softmax(outputs.logits, dim=-1)
    return pt_predictions.detach().numpy()

def get_sentiment(sentence):
  """
  Function which returns a dictionary of sentiment analysis and all entities present in sentence
  
  Parameters
  ----------
    1. sentence: str

  Returns
  -------
    1. sentiment_analysis : dict
  """

 
  # sp = spacy.load("en_core_web_md", exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
 

  # Sentence Level Sentiment:

  table = np.empty([0,7])    #creating empty table for dataframe
  senti_sent = sp(sentence)  # NER using spacy
  scores2 = None
  try:
    for span in senti_sent.sents:  # for loop on each sentance in a news
      sent = False
      for ent in span.ents:       # for loop on each entity in a sentance
        if ent.label_ == 'ORG':
          if not sent:
            scores2 = analyze_sentiment(span.text)     #Calculating scores for sentiment analysis on sentences
            table = np.append(table,np.array([[id,ent.text,ent.label_,span.text,scores2[0][1],scores2[0][0],scores2[0][2]]]),axis=0)    #appending new rows with elements
            sent=True
          else:
            table = np.append(table,np.array([[id,ent.text,ent.label_,span.text,scores2[0][1],scores2[0][0],scores2[0][2]]]),axis=0)    #appending new rows with elements

  except:
    pass

  entities = [] # list to store all the entities and its type
  unique_entity = [] # list to check unique/duplicate entity
  
  for news_sent in table:
    if news_sent[1] in unique_entity: # check for duplicate entity
      continue
    else: 
      unique_entity.append(news_sent[1]) # append if unique entity
      list_of_entity = {'entity' : news_sent [1], 'entity_type':news_sent [2]}
      entities.append(list_of_entity) 

    news_sentence = news_sent [3]
    negative_sentiment_score = news_sent [4]
    positive_sentiment_score = news_sent [5]
    neutral_sentiment_score =  news_sent [6]

  try:
    sentiment_class = get_sentiment_class(news_sent [4] ,news_sent [5] ,news_sent [6] )
    sentiment_analysis = {
              'text': news_sentence,
              'negative_sentiment_score': negative_sentiment_score,
              'positive_sentiment_score': positive_sentiment_score,
              'neutral_sentiment_score': neutral_sentiment_score,
              'sentiment_class':sentiment_class,
              'entities':entities
              }

    return sentiment_analysis

  except:
    print('No ORG present in sentence')

def compute_summary(text):
    """Compute the summary of the provided text
    """
    # var to hold the summary
    summary = ""
    # try to generate the summary
    try:
        #pre-processing the input text
        preprocess_text = text.strip().replace("\\n","")
        db_prepared_Text = "summarize: "+preprocess_text
        # print("Preprocessed text to summarization",  len(db_prepared_Text.split()))
        # tokenize the text
        tokenized_text = summarization_tokenizer.encode(db_prepared_Text, return_tensors="pt", truncation=True)
        # generating summary by using this model  
        # print("Tokenized text to summarization", tokenized_text)
        summary_db = summarization_model.generate(tokenized_text,
                                                num_beams=4,
                                                no_repeat_ngram_size=2,
                                                min_length=30,
                                                max_length=100,
                                                early_stopping=True)
        # get the summary decoded
        # print("Summary _db", summary_db)
        summary = summarization_tokenizer.decode(summary_db[0], skip_special_tokens=True)
        print("Summary :", summary)
    # if it fails, print the error
    except:
        print("Error in generating summary")
    
    # return
    return summary


def find_events(text):
    '''
    find sentences containing 'events' from provided text
    Parameters
      ----------
      text: str
          Text from which we need to find desired 'events'
    Output
      ----------
      events: list of strings
          sentences containing 'events' as a list
    '''
    # list of events
    events = []
    #pre-processing
    text_sent = event_tokenizer(str(text))

    #try to get finbert sentence embeddings
    sentences_fb = []                 #store sentence embeddings as list
    np_zeros = np.zeros((768,))

    for i in range(len(text_sent)):
      try:
        emb = finbert.sentence_vector(text_sent[i])   #emb is torch.tensor
        np_arr = emb.cpu().detach().numpy()
        sentences_fb.append(np_arr)
        #if fails to get sentence embedding, appends a numpy array of zeros 
      except:
        sentences_fb.append(np_zeros)
        print("error while getting sentence embeddings") 
    
    # model prediction. model label sentences containing events as '0' and others as '1'
    prediction_labels = event_model.predict(sentences_fb)

    # choosing 'events' from the list of sentences.
    for i in range(len(prediction_labels)):
      if prediction_labels[i] == 0:
        events.append(text_sent[i])
    # returning a list of 'events' 
    return events

class ItemList(BaseModel):
    article: List

@app.post("/items/")
async def create_item(items: ItemList):
    news_article = items.article
    sentiment_analysis_list = []
    summarization_text = []
    for item in news_article:
        sentence_sentiment = get_sentiment(item)
        
        sentiment_analysis_list.append(sentence_sentiment)
        summarization_text.append(item)

    # text summarization 
    text_article = " ".join(summarization_text)
    summarization = compute_summary(text_article)

    # event Prediction 
    events = find_events(text_article)


    # initialize empty list to store unique entities
    unique_ent = []
    # initialize empty list to store unique entities and its type
    dict_of_unique_ent = []
    for analysis in sentiment_analysis_list:
      if analysis == None:
        continue
      else:
        for object in analysis['entities']:
            # check if entity is already present in list
            if object['entity'] in unique_ent:
                continue; # continue if entity already present

            # store all unique entities and its type in dict_of_unique_ent
            else:
                unique_ent.append(object['entity'])
                entities = { 'entity' : object['entity'] , 'entity_type' : object['entity_type'] }
                dict_of_unique_ent.append(entities)

    return {"sentiment_analysis": str(sentiment_analysis_list), "ner_analysis": str(dict_of_unique_ent), "summary": str(summarization), "events": str(events)}
