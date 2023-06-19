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

#from fastapi import FastAPI
#from pydantic import BaseModel
from asyncio import events
from email.policy import default
import spacy
import pickle
from azure.cosmos import CosmosClient
import simplejson as json
import time
import numpy as np
from tqdm import tqdm
import datetime as dt
from datetime import timezone
import argparse
import ast
#import json
from decimal import Decimal
from typing import List
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM) # sentiment and summarization 

import torch.nn.functional as F
from pathlib import Path

from finbert_embedding.embedding import FinbertEmbedding        # for sentence embedding  
import en_core_web_sm 
from joblib import dump, load     # for loading trained event detection model
import nltk   # for sentence tokenization
nltk.download('punkt')

# Initializing CosmosDB Object variable
# URL = "https://fabric-database.documents.azure.com:443/"
# KEY = "H3CvHpRF3oA4R1Zjyi1vJwS5eS3XZIzayH9ShPidAFUszyeGDHI3KzDYA0auFK51Z0qFYYWeBkmYO4v3PjNTcw=="
# URL = "https://fabric-database-new.documents.azure.com:443"
# KEY = "s8Ei9XVIaLgWQCloOdWt68OXFMepK5thphRCRwOx66awsMgvYjNubgkZ9fwXXPbcKU5QRJEVPWYJACDbdDwW1w=="
# client = CosmosClient(URL, credential=KEY, consistency_level='Session')
# DATABASE_NAME = 'fabric-database-new'
# database = client.get_database_client(DATABASE_NAME)

from access_files.cosmos import run
from access_files.vault import retrieveSecrets
import asyncio
loop = asyncio.get_event_loop()
credentials = retrieveSecrets()

Temp_Config_Table = loop.run_until_complete(run(credentials['Config-Table']))
Temp_Tweets_Table = loop.run_until_complete(run(credentials['Tweet-Table']))
Temp_News_Table = loop.run_until_complete(run(credentials['News-Table']))
# Temp_Config_Table = database.get_container_client('Temp_ConfigTable')
# Temp_Tweets_Table = database.get_container_client('Temp_Tweets_Table')
# Temp_News_Table = database.get_container_client('Temp_News_Table')

print(Temp_Config_Table)
print(Temp_Tweets_Table)
print(Temp_News_Table)

# algo_access_token = "AKIA4YOVCNBK3LCH5QLO"
# algo_secret_access_token = "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"

 
NER_MODEL_NAME = 'en_core_web_md'
SENTIMENT_MODEL_NAME = 'ProsusAI/finbert'
SUMMARIZATION_MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
EVENT_DETECTION_MODEL_PATH = 'model_90.joblib'

#Command line arguments for main function
parser = argparse.ArgumentParser()
parser.add_argument('--sector', help= "Please enter name of the sector", default = "ALL")
args = parser.parse_args()


## Part: Load the models
# Model 1: Load the models for NER analysis
print(f"Loading the {NER_MODEL_NAME} model...")
# Load Spacy transformer model

#sp = en_core_web_sm.load(exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
sp = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
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
finbert = FinbertEmbedding(model_path=Path.cwd()/'fin_model')

# Model 5: Load the model for event multi class classification
print("Loading the event multi-class classification model...")
vect = 'final_linearsvc_vectorizer.pkl'
vect = pickle.load(open(vect, 'rb'))
clf = 'final_linearsvc_ec.pkl'
clf = pickle.load(open(clf, 'rb'))

# def get_ticker_names_and_id(sector):

#     config_response = []
#     ticker_names = []
#     # Enumerate the returned items
#     for item in Temp_Config_Table.query_items(
#             query='SELECT * FROM mycontainer',
#             enable_cross_partition_query=True):
#         config_response.append(item)

#     if sector == 'ALL':
#       for item in config_response:
#         ticker_names.append(item['Ticker_Name'])

#     else:
#       for item in config_response:
#         if item['Sector'] == sector:
#           ticker_names.append(item['Ticker_Name'])

#     return ticker_names
  
async def get_ticker_names_and_id(sector , table):
    """Function to get ticker_id and the ticker name
    
    Parameters:
    ----------
        sector: str
            Name of Sector - IT or BANK
        table: str 
            Name of config table - algoDDB.Table('ConfigTable')
    
    Returns:
    -------
        ticker_id: str
            ticker symbol of the company 
        ticker_name: str
            name of the company
    
    """
    ticker_name=[]
    item_list = [item async for item in table.read_all_items()]
    if sector=='ALL':
      for doc in item_list:
        ticker_name.append(doc.get('Ticker_Name'))
    else:
      for doc in item_list:
        if(doc.get('Sector')==sector):
          ticker_name.append(doc.get('Ticker_Name'))

    return ticker_name


async def fetch_data_from_table(table, ticker_names):
    this_month = (
        dt.datetime.now(timezone.utc) + dt.timedelta(hours=5, minutes=30)
    ).strftime("%Y-%m")
    today = dt.date.today()
    first = today.replace(day=1)
    lastMonth = first - dt.timedelta(days=1)
    last_month = lastMonth.strftime("%Y-%m")
    list_months = [this_month, last_month]
    print("\n list of months to be scanned ", list_months)

    print("Table :", str(table))
    if "News_Table" in str(table):
      print("\nNEWS")
    else:
      print("\nTWEETS")

    # Scanning the table for items with no NER
    # print(
    #     "\n*************scanning ",
    #     table.table_name,
    #     " table for records with no NER***********************",
    # )

    if "News_Table" in str(table):
      print("Fetching News...")
      items = []
      for name in ticker_names:
        print(name)
        print(table)

        # Enumerate the returned items
        for month in list_months:
          async for item in table.query_items(
                  query=f"SELECT * FROM c WHERE c.news_for = @name and c.news_month = @month and c.news_ner_flag = @flag ",
                  parameters=[{'name': '@name', 'value': name},
                              {'name': '@month', 'value': month},
                              {'name': '@flag', 'value': False}],
                  enable_cross_partition_query=True):
              items.append(item)

    else:
      print("Fetching Tweets...")
      items = []
      for name in ticker_names:
        print(name)
        print(table)

        # Enumerate the returned items
        for month in list_months:
          async for item in table.query_items(
                  query=f"SELECT * FROM c WHERE c.tweet_for = @name and c.tweets_month = @month and c.tweet_ner_flag = @flag",
                  parameters=[{'name': '@name', 'value': name},
                              {'name': '@month', 'value': month},
                              {'name': '@flag', 'value': False}],
                  enable_cross_partition_query=True):
              items.append(item)

    print("************ Fetched all records for ",table,"*****************************************")
    print("************ Total records fetched : ",len(items),"*********************************************")

    return items



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
    except Exception as e:
        print("Error in generating summary: ", e)
    
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
        #("np_arr: ", np_arr)
        sentences_fb.append(np_arr)
        #print("Tried")
        
        #if fails to get sentence embedding, appends a numpy array of zeros 
      except Exception as e:
        #sentences_fb.append(np_zeros)
        print("Except: ", e)
        print("error while getting sentence embeddings") 
        sentences_fb.append(np_zeros)
        
        
    if len(sentences_fb) == 0:
      print("except")
      sentences_fb.append(np_zeros)

    #print(len(sentences_fb))
    # model prediction. model label sentences containing events as '0' and others as '1'
    prediction_labels = event_model.predict(sentences_fb)

    # choosing 'events' from the list of sentences.
    for i in range(len(prediction_labels)):
      if prediction_labels[i] == 0:
        events.append(text_sent[i])
    # returning a list of 'events' 
    return events

def find_mult_class(text):

    """Take a input text sentences of 'events' and classifiy it into different pre-defined categories.
    The categories considered are: 'Debt','Dividend' ,'Financial_Results' ,'Investment_and_Funding' ,'Litigation',
            'Merger_and_Acquisition' ,'Partnership_and_JointVenture','Rating_and_Recommendation' ,
            'Securities_Securities_Turnover' and 'ShareRepurchase'
    Summary
    --------------
    Take a input text sentence and returns its most related category from the above. It returns 'Other',
    if no related categories are found.
    Parameters
    ---------------------
    text: The input sentences as a string. (eg: a paragraph of relevent event sentences)
    Returns
    --------------------
    Dictionary : Returns a dictionary of sentences (events), corresponding labels and its probability.
  
    """
    top_2_events=[]
    all_events=[]

    label_d =  {0: 'Merger_and_Acquisition',
                1: 'Partnership_and_JointVenture',
                2: 'Litigation',
                3: 'Investment_and_Funding',
                4: 'ShareRepurchase',
                5: 'Financial_Results',
                6: 'Dividend',
                7: 'Rating_and_Recommendation',
                8: 'Securities_Turnover',
                9: 'Debt',
                10: 'Other'}
                        
    events = find_events(text)

    for event in events:

        text2=str(event)
        #getting the finbert embeddings of each sentence
        print('getting sentence embeddings using tfidf vectoriser...')
        text_emb = vect.transform([text2.encode('ascii', 'ignore').decode('ascii').lower()])

        # predicting event class using pre trained LinearSVC model
        print('starting multi label classification...')
        pred = clf.predict(text_emb)

        #predicting probabilities for each class
        prob = clf._predict_proba_lr(text_emb)
 
        #converting interger labels into class labels
        labels = [label_d[np.argmax(prob)]][0]  # top label
        top_labels =[label_d[i] for i in (-prob).argsort()[0]] # all labels (descending order of their probability)

        # get probability list in descending order
        all_probabilities = list(np.sort(prob[0])[::-1])

        # To get only one class
        dict_events = {'text': event, 'labels': labels,'probability': np.max(prob)} 

        # To get top 2 classes
        top_2_events_dict = {'text': event, 'labels': top_labels[:2],'probability': all_probabilities[:2]} 
        top_2_events.append(top_2_events_dict)

        all_events_dict = {'text': event, 'labels': top_labels,'probability': all_probabilities} 
        all_events.append(all_events_dict)


    # return list of all events and it's class
    return all_events,top_2_events



def perform_enrichment(article):
    batch = nltk.tokenize.sent_tokenize(article)
    sentiment_analysis_list = []
    summarization_text = []
    for item in batch:
        sentence_sentiment = get_sentiment(item)
        sentiment_analysis_list.append(sentence_sentiment)
        #summarization_text.append(item)

    # text summarization 
    #text_article = " ".join(summarization_text)
    summarization = compute_summary(article)
    
    # event Prediction 
    #events = find_events(batch)
    start_time = time.time()
    events_detected = find_mult_class(article)
    #print("Events Detected:\n", events_detected)
    print("--- %s seconds ---" % (time.time() - start_time))

    events = {'all_events':events_detected[0],'top_2_events':events_detected[1]}

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

    output = {"sentiment_analysis": str(sentiment_analysis_list), "ner_analysis": str(dict_of_unique_ent), "summary": str(summarization), "events": str(events)}
    #count += 1
    #print("Output Events:\n", output['events'])
    return output

async def Update_table(table, record_id, record_month, Sentiment, Ner, summary, top_events, all_events):
    print("Updating table\n")
    event_flag = False
    if len(all_events) == 0:
        event_flag = False
    else:
        event_flag = True

    if "News_Table" in str(table):
        async for item in (table.query_items(query='SELECT * FROM mycontainer p WHERE p.news_month  = "{}" and p.id = "{}"'.format(record_month, record_id))):
            newItem = {
                    "id": item['id'],
                    "news_month": item['news_month'],
                    "news_ID": item['news_ID'],
                    "crawled_date": item['crawled_date'],
                    "crawled_source": item['crawled_source'],
                    "news_for": item['news_for'],  # company Name
                    "news_source": item['news_source'],  # Source of the artcile
                    "news_date": item['news_date'],  # Date of the article (datetime)
                    "link": item['link'],  # Link of the article
                    "title": item['title'],  # Title of the artcile
                    "long_description": item['long_description'],  # Content of the article
                    'short_description': item['short_description'],
                    "news_ner_flag": True,  # NER Flag for Data Enrichment Check
                    "ner_analysis": Ner,
                    "sentiment_analysis": Sentiment,
                    "summary": summary,
                    #"event_analysis": events,
                    "Top_Events": top_events,
                    "All_Events": all_events,
                    "events_flag": event_flag,
                    "events_detection_flag": True
                }

            #newItem = json.dumps(newItem, indent=True)
            #print(newItem)
            try:
                # print(newItem)
                #print(table.upsert_item(newItem))
                
                res=await table.upsert_item(newItem)
                # print(res)
                print("News inserted: ", newItem['news_ID'], "\n", newItem['news_date'])
            except Exception as e:
                print("Failed to insert news: ", e)


    else:
        async for item in (table.query_items(query='SELECT * FROM mycontainer p WHERE p.tweets_month = "{}" and p.id = "{}"'.format(record_month, record_id))):
            #print(item)
            newItem = {
                    "id": item['id'],
                    "tweets_month": item['tweets_month'],
                    "tweets_ID": item['tweets_ID'],
                    "tweet_for": item['tweet_for'],
                    "tweet_by": item['tweet_by'],
                    "tweet_date": item['tweet_date'],
                    "crawled_date": item['crawled_date'],
                    "tweet_text": item['tweet_text'],
                    "tweet_link": item['tweet_link'],
                    "tweet_ner_flag": True,
                    "ner_analysis": Ner,
                    "sentiment_analysis": Sentiment,
                    #"event_analysis": events,
                    "Top_Events": top_events,
                    "All_Events": all_events,
                    "events_flag": event_flag,
                    "events_detection_flag": True,
                }

            try:
                # print(newItem)
                #print(table.upsert_item(newItem))
                res=await table.upsert_item(newItem)
                # print(res)
                print("Tweet inserted: ", newItem['tweets_ID'], "\n", newItem['tweet_date'])
            except Exception as e:
                print("Failed to insert tweet:", e)

    print("\n........ One record Updated....... \n")



if __name__ == "__main__":
    print(args.sector)
    #ticker_id, ticker_name = get_ticker_names_and_id(args.sector)
    main_loop = asyncio.get_event_loop()
    config_names = main_loop.run_until_complete(get_ticker_names_and_id(args.sector, Temp_Config_Table))
    # print(config_names)
    tweet_items = main_loop.run_until_complete(fetch_data_from_table(Temp_Tweets_Table, config_names))
    news_items = main_loop.run_until_complete(fetch_data_from_table(Temp_News_Table, config_names))
    enriched_data_tweets = []
    enriched_data_news = []
    enriched_all_data = {}

    tweet_length = len(tweet_items)
    news_length = len(news_items)

    if len(tweet_items) != 0:
      print(tweet_items[0].keys())
      try: 
        for tweet in tweet_items:
          article = tweet['tweet_text']
          output = perform_enrichment(str(article))
          output["tweets_ID"] =  tweet['id']
          output['tweets_month'] = tweet['tweets_month']
          enriched_data_tweets.append(output)
          tweet_length = tweet_length - 1
          print("******* Remaining Tweets :",tweet_length)
      except Exception as e:
        print("Exception occured :", e)
    else:
      print("No tweets with NER = False found")

    if len(news_items) != 0:
      print(news_items[0].keys())
      try:
        for news in news_items:
          article = news['long_description']
          output = perform_enrichment(str(article))
          output["news_ID"] =  news['id']
          output['news_month'] = news['news_month']
          enriched_data_news.append(output)
          news_length = news_length - 1
          print("******** Remaining News :",news_length)
      except Exception as e:
        print("Exception occured:",e)
    print("==========================================================")
    print("==========================================================")
    print("==========================================================")
    print("==========================================================")

    enriched_all_data = {"Tweets": enriched_data_tweets, "News": enriched_data_news}
    #print(enriched_all_data)

    for item in enriched_all_data.keys():
      count = 0
      print(f"\nFor {item}\n")
      for enriched_data in enriched_all_data[item]:
        #enriched_data = output
        Sentiment = ast.literal_eval(enriched_data["sentiment_analysis"])
        Ner = ast.literal_eval(enriched_data["ner_analysis"])
        enriched_data["events"] = eval(enriched_data["events"])
        if len(enriched_data["events"]["all_events"]) == 0:
          enriched_data["events"] = {"all_events": "Null", "top_2_events": "Null"}

        #print(ast.literal_eval(enriched_data["events"]["top_2_events"]))
        
        try:
          Top_Events = enriched_data["events"]["top_2_events"]
          All_Events = enriched_data["events"]["all_events"]
          print("Top events: ", Top_Events)
        
        except Exception as e:
          print("No events detected for : ")


        #Events = ast.literal_eval(enriched_data["events"])
        formatted_sentiment = json.loads(json.dumps(Sentiment,use_decimal=True))
        formatted_ner = json.loads(json.dumps(Ner,use_decimal=True))
        #formatted_events = json.loads(json.dumps(Events), parse_float=Decimal)
        formatted_top_events = json.loads(json.dumps(Top_Events,use_decimal=True))
        formatted_all_events = json.loads(json.dumps(All_Events,use_decimal=True))
        #print("Top Events : ", formatted_top_events)
        #print(enriched_data["summary"], "\n")
        if item == "News":
          print("NEWS")
          Summary = enriched_data["summary"]
          table = Temp_News_Table
          record_id = enriched_data['news_ID']
          record_month = enriched_data['news_month']
        else:
          print("TWEET")
          Summary = []
          table = Temp_Tweets_Table
          record_id = enriched_data['tweets_ID']
          record_month = enriched_data['tweets_month']

        if (len(Sentiment) == 0) and len(Ner) == 0:
            print("\nNo enriched data received for {item} with id: ", id,"..............",)
        else:
            count += 1

        # ************ updating source table **************
        print("Entering update table")
        update_loop = asyncio.get_event_loop()
        update_loop.run_until_complete(Update_table(table, record_id, record_month, formatted_sentiment, formatted_ner, Summary, formatted_top_events, formatted_all_events))
      #print("*****************", table,"table updated with enriched data******************************")
      print("*****************Total records Updated: ", count)
      print("*****************records with no enriched data: ", (len(enriched_all_data[item]) - count))
