# THIS FILE V3 HAS THE CODE OF DISTIL BERT COMPATIBLE MODEL WITH SUMMARY MODEL AS WELL

'''
Task for NER, Sentiment Analysis, Event Detection, Classification and Summarization 

This code performs the following tasks on an article:

Recognize entities in the text (NER)
Predict sentiment for related entities [ORG]
Detect events and classify them
Generate a summary of the article

Author: 1. Shreyas R Chim <schim@alagoanalytics.com>
        2. Kalash Gandhi <kgandhi@algoanalytics.com>
        
Created v1: 3rd Dec 2021
Created v2: 7th Mar 2023
'''

import boto3
from boto3.dynamodb.conditions import Key, Attr
import spacy
import simplejson as json
import time
import numpy as np
import datetime as dt
from datetime import timezone
import argparse
import ast
from decimal import Decimal
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM) # sentiment and summarization 
import torch.nn.functional as F
from pathlib import Path
from finbert_embedding.embedding import FinbertEmbedding        # for sentence embedding  
#import en_core_web_sm 
from joblib import load     # for loading trained event detection model
import nltk   # for sentence tokenization
nltk.download('punkt')
from transformers import DistilBertTokenizer 
from transformers import TFDistilBertForSequenceClassification     #for event classification
import tensorflow as tf
import asyncio
# from aws_files import config

# AWS_KEY = config.algo_access_key
# AWS_SECRET = config.algo_secret_access_token
AWS_REGION = "ap-south-1"


# ----------------------------- SECTION- Connection to DynamoDB-------------------------------------------------
""" About function get_connection
Function to connect to AWS resource (Dynamo DB)
Input : None
Output: return connection object
"""


def get_connection():
    algoDDB = boto3.resource(
        "dynamodb",
        endpoint_url="http://dynamodb.ap-south-1.amazonaws.com",
        region_name=AWS_REGION,

        
    )
    return algoDDB


# initializing dynamodb tables containing News and Tweets
Tweets_Table = get_connection().Table("Tweets_Table")
News_Table = get_connection().Table("News_Table")
Config_Table = get_connection().Table("ConfigTable")

print(Tweets_Table)
print(type(Tweets_Table))
print(Config_Table)
print(News_Table)

# ------------------------------------------------------------------------------------------------------------------
# Config_Table = database.get_container_client('Temp_ConfigTable')
# Tweets_Table = database.get_container_client('Tweets_Table')
# News_Table = database.get_container_client('News_Table')
# algo_access_token = "AKIA4YOVCNBK3LCH5QLO"
# algo_secret_access_token = "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"

 
NER_MODEL_NAME = 'en_core_web_sm'
SENTIMENT_MODEL_NAME = 'ProsusAI/finbert'
SUMMARIZATION_MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
EVENT_DETECTION_MODEL_PATH = 'model_90.joblib'

#Command line arguments for main function
parser = argparse.ArgumentParser()
parser.add_argument('--sector', help= "Please enter name of the sector", default = "ALL")
args = parser.parse_args()
args_list=str(args.sector).split(',')
print("Printing list")
print(args_list)
## Part: Load the models
# Model 1: Load the models for NER analysis
print(f"Loading the {NER_MODEL_NAME} model...")
# Load Spacy transformer model

#sp = en_core_web_sm.load(exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
# sp = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
# sentencizer = sp.create_pipe('sentencizer')
# sp.add_pipe(sentencizer)    # add a new pipeline. It allow custom sentence boundary detection logic that doesn’t require the dependency parse.
#sp = en_core_web_sm.load(exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
try:
        sp = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
        sp.add_pipe('sentencizer')
        print("******INSIDE TRY**********")
except:
        sp = spacy.load("en_core_web_sm", exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer"])
#         sp.add_pipe(sp.create_pipe('sentencizer'))
        print("******INSIDE Except**********")

# Model 2: Load the model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME,local_files_only=True)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

# Model 3: Load the model for summarization (distilbart)
SUMMARIZATION_MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
print(f"Loading the {SUMMARIZATION_MODEL_NAME} model...")
summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)

# Model 3: Load the model for summarization (t5-small)
SUMMARIZATION_MODEL_NAME_t5 = 't5-small'
MAX_INPUT_LENGTH = 1024
print(f"Loading the {SUMMARIZATION_MODEL_NAME_t5} model...")
summarization_tokenizer_t5 = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME_t5, model_max_length=MAX_INPUT_LENGTH)
summarization_model_t5 = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME_t5)

# Model 4: Load the model for event/unknown classification
print("Loading the event detection model...")
event_tokenizer =  nltk.sent_tokenize
event_model = load(EVENT_DETECTION_MODEL_PATH)
try:
  finbert = FinbertEmbedding()
except Exception as e:
   print("ERROR: ",e)
# Model 5: Load the model for event multi class classification
print("Loading the event multi-class classification model...")
save_directory = str(Path.cwd()/'event_classification_model') # preferred location
print(save_directory)

#loading pretrained DistilBert Tokenizer and Model
tokenizer_event = DistilBertTokenizer.from_pretrained(save_directory)
model_event = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

  
async def get_ticker_names_and_id(sectors , table):
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
    for sector in sectors:
      # item_list = [item async for item in table.read_all_items()]
      # Scan the table to get all items
      response = Config_Table.scan()
      item_list = response['Items']

      if sector=='ALL':
        for doc in item_list:
          ticker_name.append(doc.get('Ticker_Name'))
      else:
        for doc in item_list:
          if(doc.get('Sector')==sector):
            ticker_name.append(doc.get('Ticker_Name'))
    print(ticker_name)
    return ticker_name


async def fetch_data_from_table(table, filter_attr):
    this_month = (
        dt.datetime.now(timezone.utc) + dt.timedelta(hours=5, minutes=30)
    ).strftime("%Y-%m")
    today = dt.date.today()
    first = today.replace(day=1)
    lastMonth = first - dt.timedelta(days=1)
    last_month = lastMonth.strftime("%Y-%m")
    list_months = [this_month, last_month]
    # list_months = ["2022-10", "2022-11"]
    print("\n list of months to be scanned ", list_months)

    if "News_Table" in str(table):
        limit=50
        key = "news_month"
    elif "Tweets_Table" in str(table):
        limit=100
        key = "tweets_month"
    # Scanning the table for items with no NER
    print(
        "*************scanning ",
        str(table),
        " table for records with no NER***********************",
    )
    items = []
    for month in list_months:
        resp_ = table.query(Limit=limit,
            KeyConditionExpression=Key(key).eq(month),
            FilterExpression=Attr(filter_attr).eq(False),
        )
        items.extend(resp_["Items"])

        while "LastEvaluatedKey" in resp_:
            try:
                resp_ = table.query(Limit=limit,
                    ExclusiveStartKey=resp_["LastEvaluatedKey"],
                    KeyConditionExpression=Key(key).eq(month),
                    FilterExpression=Attr(filter_attr).eq(False),
                )
                items.extend(resp_["Items"])
            except:
                time.sleep(5)
                resp_ = table.query(Limit=limit,
                    ExclusiveStartKey=resp_["LastEvaluatedKey"],
                    KeyConditionExpression=Key(key).eq(month),
                    FilterExpression=Attr(filter_attr).eq(False),
                )
                items.extend(resp_["Items"])    
    
    '''
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
        # Enumerate the returned items
        for month in list_months:
          async for item in table.query_items(
                  query=f"SELECT * FROM c WHERE c.tweet_for = @name and c.tweets_month = @month and c.tweet_ner_flag = @flag",
                  parameters=[{'name': '@name', 'value': name},
                              {'name': '@month', 'value': month},
                              {'name': '@flag', 'value': False}],
                  enable_cross_partition_query=True):
              items.append(item)
    '''

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
    outputs = model_sentiment(**pt_batch)
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
          print("ORG found")
          if not sent:
            scores2 = analyze_sentiment(span.text)     #Calculating scores for sentiment analysis on sentences
            table = np.append(table,np.array([[id,ent.text,ent.label_,span.text,scores2[0][1],scores2[0][0],scores2[0][2]]]),axis=0)    #appending new rows with elements
            sent=True
          else:
            table = np.append(table,np.array([[id,ent.text,ent.label_,span.text,scores2[0][1],scores2[0][0],scores2[0][2]]]),axis=0)    #appending new rows with elements

  except Exception as e:
    print('---------------- error---------------------')
    print(e)
        
  entities = [] # list to store all the entities and its type
  unique_entity = [] # list to check unique/duplicate entity
  
  for news_sent in table:
    print("news_sent: ", news_sent)
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

    print("---sentiment_analysis---\n", sentiment_analysis['entities'])
    return sentiment_analysis

  except Exception as e: 
    print(e)
    print('No ORG present in sentence')

def compute_summary(text):
    SUMMARIZATION_MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
    print(f"Loading the {SUMMARIZATION_MODEL_NAME} model...")
    summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
    
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
                                                max_length=160,
                                                early_stopping=True)
        # get the summary decoded
        # print("Summary _db", summary_db)
        summary = summarization_tokenizer.decode(summary_db[0], skip_special_tokens=True)
  
    # if it fails, print the error
    except Exception as e:
        print("Error in generating summary: ", e)
    
    # return
    return summary


def compute_summary_updated(text):

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
        tokenized_text = summarization_tokenizer_t5.encode(db_prepared_Text, max_length = 1024, return_tensors="pt", truncation=True)
        # generating summary by using this model  
        # print("Tokenized text to summarization", tokenized_text)
        summary_db = summarization_model_t5.generate(tokenized_text,
                                                num_beams=4,
                                                no_repeat_ngram_size=2,
                                                min_length=30,
                                                max_length=160,
                                                early_stopping=True)
        # get the summary decoded
        # print("Summary _db", summary_db)
        summary = summarization_tokenizer_t5.decode(summary_db[0], skip_special_tokens=True)
        summary = ' '.join([sentence.capitalize() for sentence in nltk.sent_tokenize(summary)])

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

    print("Inside EVENT FUNCTION")
    # print(text)
    # Check for empty input text or no sentences
    if not text or len(text.strip()) == 0:
      print('length of event is 0')
      return events
    else:
       print("Length of text",len(text.strip()))
    
    #pre-processing
    text_sent = event_tokenizer(str(text))
    # print('----text_sent----: ',text_sent)
    
    #try to get finbert sentence embeddings
    sentences_fb = []   #store sentence embeddings as list
    np_zeros = np.zeros((768,))
    for i in range(len(text_sent)):
        try:
            emb = finbert.sentence_vector(text_sent[i])   #emb is torch.tensor
            np_arr = emb.cpu().detach().numpy()
            sentences_fb.append(np_arr)

        except Exception as e:
            print("Except: ", e)  
            print("error while getting sentence embeddings") 
            continue  # Skip this sentence and continue with the next one
            
    if len(sentences_fb) == 0:
        print("No embeddings found")
        return events

    # model prediction. model label sentences containing events as '0' and others as '1'
    prediction_labels = event_model.predict(sentences_fb)
    # print('---prediction_labels--',prediction_labels)
    # choosing 'events' from the list of sentences.
    for i in range(len(prediction_labels)):
        if prediction_labels[i] == 0:
            events.append(text_sent[i])
    
    # returning a list of 'events' 
    return events

def find_mult_class(text):

    """Take a input text sentences of 'events' and classifiy it into different pre-defined categories.
    The categories considered are: 'Debt', 'Dividend', 'Employment', 'Financial Results' , 'Investment & Funding' , 
            'Litigation', 'Macroeconomics', 'Merger & Acquisition' , 'Partnership & JointVenture', 'Products & Services', 
            'Rating & Recommendation', 'Securities Turnover' and 'Share Repurchase'

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

    
    print("CALLING EVENT FUNCTION")                 
    events = find_events(text)
    
    label_d =  {0: 'Debt',
                1: 'Dividend',
                2: 'Employment',
                3: 'Financial Results',
                4: 'Investment & Funding',
                5: 'Litigation',
                6: 'Macroeconomics',
                7: 'Merger & Acquisition',
                8: 'Partnership & JointVenture',
                9: 'Products & Services',
                10: 'Rating & Recommendation',
                11: 'Securities Turnover',
                12: 'Share Repurchase',
                13: 'Other'}
    for event in events:

        event_text=str(event)
        #getting the finbert embeddings of each sentence
        print('Getting sentence embeddings...')
        predict_input = tokenizer_event.encode(event_text,
                                truncation=True,
                                padding=True,
                                return_tensors="tf")

        # predicting event class using pre trained DistilBert model
        print('Starting multi label classification...')
        output = model_event(predict_input)[0]
        pred=np.flip(tf.argsort(output, axis=1).numpy())[0]

        #predicting probabilities for each class
        prob=tf.nn.softmax(output[0], axis=-1).numpy()
        prob=prob.tolist()
        prob.sort(reverse=True)

        if prob[1]>=0.15 and pred[0]==13:
            prob[0],prob[1]=prob[1],prob[0]
            pred[0],pred[1]=pred[1],pred[0]
 
        #converting interger labels into class labels
        top_labels= [label_d[i] for i in pred] # all labels (descending order of their probability)

        # To get only one class
        dict_events = {'text': event_text, 'labels': top_labels[0],'probability': prob[0]} 

        # To get top 2 classes
        top_2_events_dict = {'text': event_text, 'labels': top_labels[:2],'probability': prob[:2]} 
        top_2_events.append(top_2_events_dict)

        # To get top all classes
        all_events_dict = {'text': event_text, 'labels': top_labels,'probability': prob} 
        all_events.append(all_events_dict)

        print("\n\n------top 2 events------\n", top_2_events)

    # return list of all events and it's class
    return all_events,top_2_events
  
  
def perform_enrichment(article):
    start_time1 = time.time()
    batch = nltk.tokenize.sent_tokenize(article)
    print("Tokenize time --- %s seconds ---" % (time.time() - start_time1))
    sentiment_analysis_list = []
    summarization_text = []
    start_time2 = time.time()
    for item in batch:
        sentence_sentiment = get_sentiment(item)
        sentiment_analysis_list.append(sentence_sentiment)
        #summarization_text.append(item)
    print("Sentiment time --- %s seconds ---" % (time.time() - start_time2))

    # text summarization 
    #text_article = " ".join(summarization_text)
    start_time3 = time.time()
    try:
      summarization = compute_summary_updated(article) #computing summary using T5 small
    except:
      summarization = compute_summary(article) #computing summary using Distilbart in case T5 small doesn't work
    print("Summary time --- %s seconds ---" % (time.time() - start_time3))
    # event Prediction 
    #events = find_events(batch)
    start_time4 = time.time()
    events_detected = find_mult_class(article)
    #print("Events Detected:\n", events_detected)
    print("Muliclass time --- %s seconds ---" % (time.time() - start_time4))

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

async def update_table(table, record_id, record_month, Sentiment, Ner, summary, top_events, all_events):
    print("Updating table\n")
    count = 0
    event_flag = False
    if len(all_events) == 0:
        event_flag = False
    else:
        event_flag = True

    if table.table_name == "Tweets_Table":
        try:
          table.update_item(
              Key={"tweets_month": record_month, "tweets_ID": record_id},
              UpdateExpression='''set tweet_ner_flag = :flag, ner_analysis = :ner, 
              sentiment_analysis = :sentiment, Top_Events = :Top_Events, All_Events = :All_Events, 
              events_flag = :events_flag, events_detection_flag = :events_detection_flag''',
              ExpressionAttributeValues={
                  ":flag": True,
                  ":ner": Ner,
                  ":sentiment": Sentiment,
                  # ":events": events,
                  ":Top_Events": top_events,
                  ":All_Events": all_events,
                  ":events_flag": event_flag,
                  ":events_detection_flag": True,
              },
          )
          count += 1

        except Exception as e:
           print("Error in Entering Tweet: ", e)

    if table.table_name == "News_Table":
        try:
          table.update_item(
              Key={"news_month": record_month, "news_ID": record_id},
              UpdateExpression='''set news_ner_flag = :flag , ner_analysis= :ner , 
              sentiment_analysis = :sentiment, summary = :summary, Top_Events = :Top_Events, All_Events = :All_Events,  
              events_flag = :events_flag, events_detection_flag = :events_detection_flag''',
              ExpressionAttributeValues={
                  ":flag": True,
                  ":ner": Ner,
                  ":sentiment": Sentiment,
                  ":summary": summary,
                  # ":events": events,
                  ":Top_Events": top_events,
                  ":All_Events": all_events,
                  ":events_flag": event_flag,
                  ":events_detection_flag": True,
              },
          )
          count += 1
        except Exception as e:
           print("Error in Entering News: ", e)

#     if "News_Table" in str(table):
#         async for item in (table.query_items(query='SELECT * FROM mycontainer p WHERE p.news_month  = "{}" and p.id = "{}"'.format(record_month, record_id))):
#             newItem = {
#                     # "id": item['id'],
#                     "news_month": item['news_month'],
#                     "news_ID": item['news_ID'],
#                     "crawled_date": item['crawled_date'],
#                     "crawled_source": item['crawled_source'],
#                     "news_for": item['news_for'],  # company Name
#                     "news_source": item['news_source'],  # Source of the artcile
#                     "news_date": item['news_date'],  # Date of the article (datetime)
#                     "link": item['link'],  # Link of the article
#                     "title": item['title'],  # Title of the artcile
#                     "long_description": item['long_description'],  # Content of the article
#                     'short_description': item['short_description'],
#                     "news_ner_flag": True,  # NER Flag for Data Enrichment Check
#                     "ner_analysis": Ner,
#                     "sentiment_analysis": Sentiment,
#                     "summary": summary,
#                     #"event_analysis": events,
#                     "Top_Events": top_events,
#                     "All_Events": all_events,
#                     "events_flag": event_flag,
#                     "events_detection_flag": True
#                 }

#             #newItem = json.dumps(newItem, indent=True)
#             #print(newItem)
#             try:
#                 # print(newItem)
#                 #print(table.upsert_item(newItem))
                
#                 res=await table.upsert_item(newItem)
#                 # print(res)
# #                 print("News inserted: ", newItem['news_ID'], "\n", newItem['news_date'])
#             except Exception as e:
#                 print("Failed to insert news: ", e)


#     else:
#         async for item in (table.query_items(query='SELECT * FROM mycontainer p WHERE p.tweets_month = "{}" and p.id = "{}"'.format(record_month, record_id))):
#             #print(item)
#             newItem = {
#                     # "id": item['id'],
#                     "tweets_month": item['tweets_month'],
#                     "tweets_ID": item['tweets_ID'],
#                     "tweet_for": item['tweet_for'],
#                     "tweet_by": item['tweet_by'],
#                     "tweet_date": item['tweet_date'],
#                     "crawled_date": item['crawled_date'],
#                     "tweet_text": item['tweet_text'],
#                     "tweet_link": item['tweet_link'],
#                     "tweet_ner_flag": True,
#                     "ner_analysis": Ner,
#                     "sentiment_analysis": Sentiment,
#                     #"event_analysis": events,
#                     "Top_Events": top_events,
#                     "All_Events": all_events,
#                     "events_flag": event_flag,
#                     "events_detection_flag": True,
#                 }

#             try:
#                 # print(newItem)
#                 #print(table.upsert_item(newItem))
#                 res=await table.upsert_item(newItem)
#                 # print(res)
# #                 print("Tweet inserted: ", newItem['tweets_ID'], "\n", newItem['tweet_date'])
#             except Exception as e:
#                 print("Failed to insert tweet:", e)
            
    print("\n........ One record Updated....... \n")
    return count

def upload_data(item,enriched_data):
    # count = 0
    Sentiment = ast.literal_eval(enriched_data["sentiment_analysis"])
    Ner = ast.literal_eval(enriched_data["ner_analysis"])
    enriched_data["events"] = eval(enriched_data["events"])

    if len(enriched_data["events"]["all_events"]) == 0:
      enriched_data["events"] = {"all_events": "Null", "top_2_events": "Null"}

    try:
      Top_Events = enriched_data["events"]["top_2_events"]
      All_Events = enriched_data["events"]["all_events"]
    
    except Exception as e:
      print("No events detected for : ")

    formatted_sentiment = json.loads(json.dumps(Sentiment), parse_float=Decimal)
    formatted_ner = json.loads(json.dumps(Ner), parse_float=Decimal)
    formatted_top_events = json.loads(json.dumps(Top_Events), parse_float=Decimal)
    formatted_all_events = json.loads(json.dumps(All_Events), parse_float=Decimal)

    if item == "News":
      print("NEWS")
      Summary = enriched_data["summary"]
      table = News_Table
      record_id = enriched_data['news_ID']
      record_month = enriched_data['news_month']
    else:
      print("TWEET")
      Summary = []
      table = Tweets_Table
      record_id = enriched_data['tweets_ID']
      record_month = enriched_data['tweets_month']

    if (len(Sentiment) == 0) and len(Ner) == 0:
        print(f"\nNo enriched data received for {item} with id: {record_id}..............",)
    else:
        # count += 1
        pass

    print("Entering update table")
    update_loop = asyncio.get_event_loop()
    count = update_loop.run_until_complete(update_table(table, record_id, record_month, formatted_sentiment, formatted_ner, Summary, formatted_top_events, formatted_all_events))
    print("*****************", table,"table updated with enriched data******************************")
    print("Count: ", count)
    return count


if __name__ == "__main__":
    print(args_list)
    #ticker_id, ticker_name = get_ticker_names_and_id(args.sector)
    main_loop = asyncio.get_event_loop()
    config_names = main_loop.run_until_complete(get_ticker_names_and_id(args_list, Config_Table))
    # print(config_names)
    # tweet_items = main_loop.run_until_complete(fetch_data_from_table(Tweets_Table, config_names))
    # news_items = main_loop.run_until_complete(fetch_data_from_table(News_Table, config_names))
    # ******************************************** TWEETS TABLE starting point ***********************************************
    # fetch items from tweets_table
    tweet_items = main_loop.run_until_complete(fetch_data_from_table(Tweets_Table, "tweet_ner_flag"))
    # if no items are found with 'tweet_ner_flag'=False then break the code since all items are updated with enriched data

    # ******************************************** NEWS TABLE starting point ***********************************************

    # fetch items from news_table
    news_items = main_loop.run_until_complete(fetch_data_from_table(News_Table, "news_ner_flag"))
    # if no items are found with  'ner' =False then break the code since all items are updated with enriched data

    enriched_data_tweets = []
    enriched_data_news = []
    enriched_all_data = {}

    tweet_length = len(tweet_items)
    news_length = len(news_items)
    
    tot_enrich=0
    not_enrich=0
    if len(tweet_items) != 0:
#       print(tweet_items[0].keys())
      try: 
        for tweet in tweet_items:
          article = tweet['tweet_text']
          output = perform_enrichment(str(article))
          output["tweets_ID"] =  tweet['tweets_ID']
          output['tweets_month'] = tweet['tweets_month']
          if upload_data("Tweets",output) == 0:
            not_enrich+=1
          else :
            tot_enrich+=1
        #   enriched_data_tweets.append(output)
          tweet_length = tweet_length - 1
          print("******* Remaining Tweets :",tweet_length)
      except Exception as e:
        print("Exception occured :", e)
    else:
      print("No tweets with NER = False found")

    if len(news_items) != 0:
#       print(news_items[0].keys())
      try:
        for news in news_items:
          article = news['long_description']
          output = perform_enrichment(str(article))
          output["news_ID"] =  news['news_ID']
          output['news_month'] = news['news_month']
          if upload_data("News",output) == 0:
            not_enrich+=1
          else :
            tot_enrich+=1
        #   enriched_data_news.append(output)
          news_length = news_length - 1
          print("******** Remaining News :",news_length)
      except Exception as e:
        print("Exception occured:",e)

    print("==========================================================")


    # enriched_all_data = {"Tweets": enriched_data_tweets, "News": enriched_data_news}

    print("*****************Total records Updated: ", tot_enrich)
    print("*****************records with no enriched data: ", not_enrich)
    
