import os
import time
import datetime
from typing import Dict

import uvicorn
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware


import logging

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.basicConfig(
        filename="logs/Semantic_Search_" + datetime.date.today().strftime("%Y-%m-%d").replace('-','_').replace(':','_') + ".log",
        level=logging.INFO,
        filemode='w+',
        format="%(asctime)s:%(levelname)s:%(message)s"
    )

# Set logging level of Haystack to warning
logging.getLogger("haystack").setLevel(logging.WARNING)

# Set logging level of Azure to Warning to avoid unnecessary HTTP logs
Azure_logger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
Azure_logger.setLevel(logging.WARNING)


from models import load_document_search_pipeline, load_query_classifier, load_reader
from utils import get_dynamodb_connection, get_dynamodb_table, process_prediction, process_qna_prediction

from dotenv import load_dotenv
load_dotenv()

start_time = time.time()

# print("***************Start_time:"+str(start_time)+"*****************")

app = FastAPI(title="Semantic Search API",
            contact={
                "name": "AlgoAnalytics Pvt Ltd",
                "url": "https://algoanalytics.com/",
                "email": "info@algoanalytics.com"
            })

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

dynamodb = None 
news_table = tweets_table = None
news_pipeline = tweets_pipeline = None 
query_classifier = reader = None

@app.on_event('startup')
def init_data():
    """Initialize resources. Loads DynamoDB, Document Search Pipelines, Query Classifier and Reader.
    """
    logging.info("The application is initialising data. Please wait.")
    global dynamodb, news_table, tweets_table, news_pipeline, tweets_pipeline, query_classifier, reader

    logging.info("Connecting to DynamoDB and loading news and tweets table")
    dynamodb = get_dynamodb_connection()
    news_table = get_dynamodb_table(dynamodb, type="news")
    tweets_table = get_dynamodb_table(dynamodb, type="tweets")

    logging.info("Loading news and tweets search pipeline")
    news_pipeline = load_document_search_pipeline(type="news")
    tweets_pipeline = load_document_search_pipeline(type="tweets")

    logging.info("Loading query classifier and haystack reader")
    query_classifier = load_query_classifier()
    reader = load_reader()


@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """Returns HTTP 200 is the App is running"""
    logging.info("The app is running!")
    return {"message" : "The app is running!"}


@app.get("/search/{search_string}", response_model=dict)
def get_search_results(search_string: str):
    """Search for query string, and return the top 10 results.

    Args:
        search_string (str): The query string to search for.

    Returns:
        dict: A dictionary containing news result and tweets result.
    """
    global start_time, news_pipeline, tweets_pipeline, query_classifier, reader
    
    log_time = datetime.datetime.now()
    
    # Get last_modified_time of retrievers
    last_modified_time_news = os.path.getmtime(os.path.join(os.environ['DOCUMENT_STORE_DIR'], "news_faiss_index.faiss"))
    last_modified_time_tweets = os.path.getmtime(os.path.join(os.environ['DOCUMENT_STORE_DIR'], "tweets_faiss_index.faiss"))

    
    if last_modified_time_news > start_time:
        logging.info("Loading newly trained haystack news model. Please wait.")

        news_pipeline = load_document_search_pipeline(type="news")
        start_time = last_modified_time_news

    if last_modified_time_tweets > start_time:
        logging.info("Loading newly trained haystack tweets model. Please wait.")

        tweets_pipeline = load_document_search_pipeline(type="tweets")
        start_time = last_modified_time_tweets
    
    logging.info("New search string received : %r", search_string)

    logging.info("Running news search pipeline on the search string. Please wait.")
    news_pred = news_pipeline.run(query=search_string)

    logging.info("Running tweets search pipeline on the search string. Please wait.")
    tweets_pred = tweets_pipeline.run(query=search_string)
    
    logging.info("Processing news and tweets pipeline response. Please wait.")
    news_search_results, news_scores, news = process_prediction(news_pred, news_table, type="news")
    tweets_search_results, tweets_scores, tweets = process_prediction(tweets_pred, tweets_table, type="tweets")

    
    # Check if search results are returned
    if not news_search_results:
        logging.error("No news search results found!")
        raise HTTPException(status_code=204, detail="Could not fetch news search results")

    if not tweets_search_results:
        logging.error("No news search results found!")
        raise HTTPException(status_code=204, detail="Could not fetch tweets search results")


    # Condition to check whether QnA is to be included.
    # If query type is Question, execute the QnA Reader component.
    logging.info("Checking if the search string has a question.")
    qna_present = (query_classifier.run(search_string)[1] == 'output_1')
    
    if qna_present:
        logging.info("Found a question in the search string. Running QnA pipeline now.")
        # all_documents = news_pred['documents'] + tweets_pred['documents']
        
        # Only send news documents for QnA prediction now
        all_documents = news_pred['documents']
        qna_pred = reader.predict(search_string, all_documents)
        
        qna_results, qna_scores = process_qna_prediction(qna_pred, news_pred, news_search_results, news)
        
    news_result = {
        "search_string": search_string, 
        "search_result": news_search_results,
        "score": news_scores,
        "news": news,
        "timestamp": log_time
    }
    
    tweets_result = {
        "search_string": search_string,
        "search_result": tweets_search_results,
        "score": tweets_scores,
        "tweets": tweets,
        "timestamp": log_time
    }
    
    total_results = {
        'News_Search_Results': news_result ,
        'Tweets_Search_Results': tweets_result
    } 

    if qna_present:
        qna_total_result = {
            'All_QnA_Results': qna_results,
            'All_QnA_Scores': qna_scores
        }
        
        total_results['QnA_Results'] = qna_total_result
    
    logging.info("Semantic Search Prediction complete. Returning API response now.")
    logging.info("Returning API response now.\n%r", total_results)
    return total_results


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)
