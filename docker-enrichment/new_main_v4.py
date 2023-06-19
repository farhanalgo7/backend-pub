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
from spacy.language import Language
#from aws_files import config
from nltk import sent_tokenize
from fastcoref import spacy_component
from spacy.tokens import Doc, Span


nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("fastcoref")

#AWS_KEY = config.algo_access_key
#AWS_SECRET = config.algo_secret_access_token
AWS_REGION = "ap-south-1"


# ----------------------------- SECTION- Connection to DynamoDB-------------------------------------------------

def get_connection():
    algoDDB = boto3.resource(
        "dynamodb",
        #aws_access_key_id=AWS_KEY ,
        endpoint_url="http://dynamodb.ap-south-1.amazonaws.com",
        #aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,

        
    )
    return algoDDB


# initializing dynamodb tables containing News
News_Table = get_connection().Table("News_Table")
Config_Table = get_connection().Table("ConfigTable")

NER_MODEL_NAME = 'en_core_web_sm'
SENTIMENT_MODEL_NAME = 'ProsusAI/finbert'
SUMMARIZATION_MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
EVENT_DETECTION_MODEL_PATH = 'event_classification_model/model_90.joblib'

#Command line arguments for main function
parser = argparse.ArgumentParser()
parser.add_argument('--sector', help= "Please enter name of the sector", default = "ALL")
args = parser.parse_args()
args_list=str(args.sector).split(',')

## Part: Load the models

# Model 1: Load the model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

# Model 2: Load the model for summarization (distilbart)
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

#loading pretrained DistilBert Tokenizer and Model
tokenizer_event = DistilBertTokenizer.from_pretrained(save_directory)
model_event = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

  
async def get_ticker_names_and_id(sectors , table):
    """
    Function to get ticker_id and the ticker name
    Parameters:
    ----------
        sectors: str
            Name of Sector - IT or BANK
        table: str 
            Name of config table - algoDDB.Table('ConfigTable')
    
    Returns:
    -------
        ticker_name: str
            name of the company
    """

    ticker_name=[]
    for sector in sectors:
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

    if "News_Table" in str(table):
        limit=50
        key = "news_month"
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
    

    print("************ Fetched all records for *****************************************")
    print("************ Total records fetched : ",len(items),"*********************************************")

    return items


def get_sentiment_class(neg, pos, neut):
    if (pos> neg) and (pos > neut):
      sentiment_class = 'positive'
    elif (neg > pos) and (neg > neut):
      sentiment_class = 'negative'
    elif (neut> neg) and (neut > pos):
      sentiment_class = 'neutral' 
    return sentiment_class

def get_sentiment_modified(sentence):
  """
  This modified function does not perform NER since the entity which we are focusing
  on is the same company to which the news belongs and that has been passed as 
  TSA_keyword.

  Parameters
      -------------
      1. sentence: str
  Returns
  -------
    1. sentiment_analysis : dict
  """
  predictions_np = analyze_sentiment(sentence)
  # Calculate sentiment counts
  pos = np.sum(predictions_np[:, 0]) 
  neg = np.sum(predictions_np[:, 1])
  neut = np.sum(predictions_np[:, 2])

  sentiment_class = get_sentiment_class(neg, pos, neut)

  sentiment_analysis = {
            'text': sentence,
            'negative_sentiment_score': neg,
            'positive_sentiment_score': pos,
            'neutral_sentiment_score': neut,
            'sentiment_class':sentiment_class
            }

  return sentiment_analysis

def analyze_sentiment(doc):
    """ Function to perform Sentiment analysis using ProsusAI/finbert tokenizer and pretrained NLP model.
    
    The function will take a doc/text as input and first apply tokenizer and then calculate outputs by using model function on it. 
    We will perform softmax on outputs and return numpy array of pt_predictions.
    Parameters
      -------------
            1. doc : a single string
      Returns
      ----------
        pt_predictions.detach().numpy(): numpy array of pt_predictions 
    """
    pt_batch = tokenizer(doc,padding=True,truncation=True,max_length=512,return_tensors="pt")
    outputs = model_sentiment(**pt_batch)
    pt_predictions = F.softmax(outputs.logits, dim=-1)

    return pt_predictions.detach().numpy()


def extract_org_frequency(article):
    """
    Extracts organizations (ORG entities) from the given article and returns their frequency.

    Args:
        article (str): The input article to extract organizations from.

    Returns:
        list: A list of dictionaries, each containing the 'entity' (organization) and 'frequency' keys.

    Example:
        article = "An article about technology companies such as Google, Apple, and Microsoft."
        result = extract_org_frequency(article)
        print(result)
        Output: [{'entity': 'Google', 'frequency': 1}, {'entity': 'Apple', 'frequency': 1},
                 {'entity': 'Microsoft', 'frequency': 1}]
    """
    doc = nlp(article)
    org_frequency = {}
    
    for entity in doc.ents:
        if entity.label_ == 'ORG':
            org = entity.text
            org_frequency[org] = org_frequency.get(org, 0) + 1
    
    result = [{'entity': org, 'frequency': freq} for org, freq in org_frequency.items()]
    return result

def compute_summary(text):
    """
    Compute the summary of the provided text
    """
    SUMMARIZATION_MODEL_NAME = 'sshleifer/distilbart-cnn-12-6'
    print(f"Loading the {SUMMARIZATION_MODEL_NAME} model...")
    summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
    
    summary = ""
    try:
        #pre-processing the input text
        preprocess_text = text.strip().replace("\\n","")
        db_prepared_Text = "summarize: "+preprocess_text
        # tokenize the text
        tokenized_text = summarization_tokenizer.encode(db_prepared_Text, return_tensors="pt", truncation=True)
        # generating summary by using this model  
        summary_db = summarization_model.generate(tokenized_text,
                                                num_beams=4,
                                                no_repeat_ngram_size=2,
                                                min_length=30,
                                                max_length=160,
                                                early_stopping=True)
        # get the summary decoded
        summary = summarization_tokenizer.decode(summary_db[0], skip_special_tokens=True)
    except Exception as e:
        print("Error in generating summary: ", e)

    return summary

def compute_summary_updated(text):
    """
    Compute the summary of the provided text
    """
    summary = ""
    try:
        #pre-processing the input text
        preprocess_text = text.strip().replace("\\n","")
        db_prepared_Text = "summarize: "+preprocess_text
        # tokenize the text
        tokenized_text = summarization_tokenizer_t5.encode(db_prepared_Text, max_length = 1024, return_tensors="pt", truncation=True)
        # generating summary by using this model  
        summary_db = summarization_model_t5.generate(tokenized_text,
                                                num_beams=4,
                                                no_repeat_ngram_size=2,
                                                min_length=30,
                                                max_length=160,
                                                early_stopping=True)
        # get the summary decoded
        summary = summarization_tokenizer_t5.decode(summary_db[0], skip_special_tokens=True)
        summary = ' '.join([sentence.capitalize() for sentence in nltk.sent_tokenize(summary)])
    except Exception as e:
        print("Error in generating summary: ", e)
    
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
    events = []

    # Check for empty input text or no sentences
    if not text or len(text.strip()) == 0:
      print('length of event is 0')
      return events
    else:
       print()
    
    #pre-processing
    text_sent = event_tokenizer(str(text))
    
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
    # choosing 'events' from the list of sentences.
    for i in range(len(prediction_labels)):
        if prediction_labels[i] == 0:
            events.append(text_sent[i])

    return events

def find_mult_class(text):

    """
    Take a input text sentences of 'events' and classifiy it into different pre-defined categories.
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
        predict_input = tokenizer_event.encode(event_text,
                                truncation=True,
                                padding=True,
                                return_tensors="tf")

        # predicting event class using pre trained DistilBert model
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

        # To get top 2 classes
        top_2_events_dict = {'text': event_text, 'labels': top_labels[:2],'probability': prob[:2]} 
        top_2_events.append(top_2_events_dict)

        # To get top all classes
        all_events_dict = {'text': event_text, 'labels': top_labels,'probability': prob} 
        all_events.append(all_events_dict)

    return all_events,top_2_events

def extract_sents(text, company_name): # Pass the news article and company name in lowercase to this function. It returns the imp lines of news as a list
  """
  This function is extracting sentences addressing the target company using coreferencing
  """
  
  doc = nlp(text)

  # Get the coreference clusters from the processed text
  clusters = doc._.coref_clusters


  for cluster in clusters:
      mentions = [text[start:end] for start, end in cluster]


  final_cluster = []

  for cluster in clusters:
      mentions = [text[start:end] for start, end in cluster]
      for element in mentions:
        if (company_name.lower() in element.lower()):
          target = mentions
          final_cluster = cluster

  if(len(final_cluster)==0):
      return []
          

  # Split the text into sentences
  sentences = sent_tokenize(text)

  start_pos = 0
  sentiment_sentences=[]
  for i, sentence in enumerate(sentences):
      # Calculate the ending position of the current sentence
      end_pos = start_pos + len(sentence)
      
      try:
        for position in final_cluster:
            # Check if the starting position of the current coreference falls within the current sentence
            if (position[0] >= start_pos) and (position[0] <= end_pos):
                sentiment_sentences.append(sentence)
                break

      except:
        print('Company keyword not found in sentence')

      # Update the starting position for the next sentence
      start_pos = end_pos + 2 # add 2 to account for the '. ' separator

  return sentiment_sentences

def perform_enrichment(article, tsa_keyword):
    #############################################
    print('PERFORMING NER')
    start_time1 = time.time()
    ner_org_analysis = extract_org_frequency(article)

    #############################################
    print('PERFORMING Targeted Sentiment Analysis')
    start_time2 = time.time()
    batch = extract_sents(article, tsa_keyword)
    print("Tokenize time --- %s seconds ---" % (time.time() - start_time1))
    sentiment_analysis_list = []
    for item in batch:
        sentence_sentiment = get_sentiment_modified(item)
        sentiment_analysis_list.append(sentence_sentiment)
    print("Sentiment time --- %s seconds ---" % (time.time() - start_time2))

    #############################################
    print('PERFORMING Summarization')
    start_time3 = time.time()
    try:
      summarization = compute_summary_updated(article) #computing summary using T5 small
    except:
      summarization = compute_summary(article) #computing summary using Distilbart in case T5 small doesn't work
    print("Summary time --- %s seconds ---" % (time.time() - start_time3))

    #############################################
    print('PERFORMING Event Detection and Classification')
    start_time4 = time.time()
    events_detected = find_mult_class(article)
    print("Muliclass time --- %s seconds ---" % (time.time() - start_time4))
    events = {'all_events':events_detected[0],'top_2_events':events_detected[1]}

    #############################################
    perform_enrichment_output = {"sentiment_analysis": str(sentiment_analysis_list),"ner_analysis": str(ner_org_analysis), 
                                 "summary": str(summarization), "events": str(events)}
    

    print('THE OUTPUT of The article is:')
    print(perform_enrichment_output)
    return perform_enrichment_output

async def update_table(table, record_id, record_month, Sentiment, summary, top_events, all_events):
    print("Updating table\n")
    count = 0
    event_flag = False
    if len(all_events) == 0:
        event_flag = False
    else:
        event_flag = True

    if table.table_name == "News_Table":
        try:
          table.update_item(
              Key={"news_month": record_month, "news_ID": record_id},
              UpdateExpression='''set news_ner_flag = :flag , 
              sentiment_analysis = :sentiment, summary = :summary, Top_Events = :Top_Events, All_Events = :All_Events,  
              events_flag = :events_flag, events_detection_flag = :events_detection_flag''',
              ExpressionAttributeValues={
                  ":flag": True,
                  # ":ner": Ner,
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
            
    print("\n........ One record Updated....... \n")
    return count

async def fetch_config_data(table):
    """
    Args:
        table (boto3.resource): The DynamoDB table object.

    Returns:
        list: A list of all the items in the table.
    """
    items = []
    response = table.scan()
    items.extend(response['Items'])
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])
    return items


def upload_data(item,enriched_data):
    Sentiment = ast.literal_eval(enriched_data["sentiment_analysis"])
    enriched_data["events"] = eval(enriched_data["events"])

    if len(enriched_data["events"]["all_events"]) == 0:
      enriched_data["events"] = {"all_events": "Null", "top_2_events": "Null"}

    try:
      Top_Events = enriched_data["events"]["top_2_events"]
      All_Events = enriched_data["events"]["all_events"]
    except Exception as e:
      print("No events detected for : ")

    formatted_sentiment = json.loads(json.dumps(Sentiment), parse_float=Decimal)
    formatted_top_events = json.loads(json.dumps(Top_Events), parse_float=Decimal)
    formatted_all_events = json.loads(json.dumps(All_Events), parse_float=Decimal)

    if item == "News":
      print("NEWS")
      Summary = enriched_data["summary"]
      table = News_Table
      record_id = enriched_data['news_ID']
      record_month = enriched_data['news_month']

    if len(Sentiment) == 0:
        print(f"\nNo enriched data received for {item} with id: {record_id}..............",)
    else:
        # count += 1
        pass

    update_loop = asyncio.get_event_loop()
    count = update_loop.run_until_complete(update_table(table, record_id, record_month, formatted_sentiment, Summary, formatted_top_events, formatted_all_events))
    print("*****************table updated with enriched data******************************")
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
    # ******************************************** NEWS TABLE starting point ***********************************************
    # fetch items from news_table
    news_items = main_loop.run_until_complete(fetch_data_from_table(News_Table, "news_ner_flag"))
    # if no items are found with  'ner' =False then break the code since all items are updated with enriched data
    config_items = main_loop.run_until_complete(fetch_config_data(Config_Table))

    enriched_data_news = []
    enriched_all_data = {}

    news_length = len(news_items)
    
    tot_enrich=0
    not_enrich=0

    if len(news_items) != 0:
      try:
        company_keywords = {d['Ticker_Name']: d['TSA_keywords'] for d in config_items}
        for news in news_items:  
          article = news['long_description']
          companyname = news['news_for']
          keyword_tsa = company_keywords.get(companyname)
          try:
            output = perform_enrichment(str(article), keyword_tsa)
            output["news_ID"] =  news['news_ID']
            output['news_month'] = news['news_month']
          except Exception as e:
            print("Exception occured in perform enrichment:",e)
          if upload_data("News",output) == 0:
             not_enrich+=1
          else :
             tot_enrich+=1
           #   enriched_data_news.append(output)
          news_length = news_length - 1
          print("******** Remaining News :",news_length)
  
      except Exception as e:
        print("Exception occured in company keywords:",e)

    print("*****************Total records Updated: ", tot_enrich)
    print("*****************records with no enriched data: ", not_enrich)
