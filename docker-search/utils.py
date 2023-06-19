import os
import json
import datetime as dt
import logging

import boto3
from boto3.dynamodb.conditions import Key, Attr

from dotenv import load_dotenv
load_dotenv()

def check_type(type):
    """Check if the type is "news" or "tweets". If not, raise error.

    Args:
        type (str): String specifying "news" or "tweets"

    Raises:
        ValueError: If type is not "news" or "tweets"
    """
    if type not in ["news", "tweets"]:
        raise ValueError("Invalid type. Must be 'news' or 'tweets'.")
        logging.error("Wrong type detected. Must be news or tweets.")


def get_dynamodb_connection():
    """Get DynamoDB Connection

    Returns:
        dynamodb.ServiceResource: Connection to DynamoDB
    """
    try:
        algoDDB = boto3.resource(
            "dynamodb",
            aws_access_key_id=os.environ["AWS_TOKEN"],
            # endpoint_url=os.environ['AWS_ENDPOINT'],
            aws_secret_access_key=os.environ["AWS_KEY"],
            region_name=os.environ["AWS_REGION"]

        )
        logging.info("Connected to AWS DynamoDB")
        return algoDDB
    
    except Exception as e:
        logging.error("Could not connect to AWS DynamoDB.")
        logging.error(f"Exception:\n{e}")
        print(e)

def get_dynamodb_table(ddb_connection, type="news"):
    """Get DynamoDB Table using the connection for a given type of "news" or "tweets"

    Args:
        ddb_connection (dynamodb.ServiceResource): Connection to DynamoDB
        type (str, optional): Type of "news" or "tweets". Defaults to "news".

    Returns:
        dynamodb.Table: DynamoDB Table, either "News_Table" or "Tweets_Table", depending on type
    """
    check_type(type)

    try:
        # The DynamoDB Tables are named 'News_Table' and 'Tweets_Table'
        # So need to convert to title-case
        logging.info(f"Obtained DynamoDB Table for {type.title()}")
        return ddb_connection.Table(f"{type.title()}_Table")
    
    except Exception as e:
        logging.error(f"Could not get DynamoDB Table for {type.title()}")
        logging.error(f"Exception:\n{e}")
        print(e)


def process_prediction(pred, container, type="news"):
    """Process prediction given by the DocumentSearch pipeline. 3 things are returned.
    - search_results contains just the news/tweet content.
    - scores contains sentiment score for the news/tweets.
    - documents contains the entire information of news/tweets along with metadata.

    Args:
        pred (dict): DocumentSearch Pipeline's output for a given query
        container (container.ContainerProxy): Cosmos Container, required for getting document entries using UUIDs
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".

    Returns:
        dict, dict, dict: Search Results, Scores and Documents (which contains all info along with metadata)
    """
    check_type(type)
    
    try:
        # Create a dictionary of the form f"content{i}_source_name:document_name" for each result in the prediction
        # Document_names look like f"doc_{UUID}.txt", for example "doc_743c605d-4fa3-4656-aefb-a03dbfef65d8.txt"
        source_names = {f"content{i}_source_name":pred['documents'][i].meta.get('name', '') for i in range(len(pred['documents']))}
        
        if len(source_names) < 1:
            logging.warning(f"Could not get document filenames from prediction for {type}.")

        # Function to get formatted document contents from AWS DynamoDB
        documents = get_documents_from_filenames(source_names, container, type=type)
        
        if len(documents) < 1:
            logging.warning(f"Could not get documents from filenames for {type}.")
        
        # Search Results only have news/tweet content, without any other information
        # documents.keys() look like content0, content1, and so on.
        search_results = {i:pred["documents"][int(i.replace("content",""))].content for i in documents.keys()}

        # Get sentiment scores for each of the documents
        scores = {f"{i}_score":round(pred["documents"][int(i.replace("content",""))].score, 5) for i in documents.keys()}

        # Replace the search_result with the document summary, if it exists
        for i in range(len(search_results)):
            summary = documents[f"content{i}"][f"{type.title()}_Summary"]

            if summary != "":
                search_results[f"content{i}"] = summary

        return search_results, scores, documents
    
    except KeyError as ke:
        logging.error(f"Key Error in some dictionary, possibly in prediction. Either it has no documents, or the metadata does not have names.")
        logging.error(f"Exception:\n{ke}")
        print(ke)
    
    except Exception as e:
        logging.error(f"Something went wrong in Processing Predictions for {type}.")
        logging.error(f"Exception:\n{e}")
        print(e)
        
        return {}, {}, {}


def get_documents_from_filenames(file_names, container, type="news"):
    """Get formatted document entries for front-end from their filenames. It first creates a list of UUIDs,
    then gets Document IDs from UUIDs, then gets the documents from DynamoDB using these document IDs, and 
    then formats for front-end, all using helper functions.

    Args:
        file_names (list): List of UUID
        container (container.ContainerProxy): Cosmos Container, either "News_Table" or "Tweets_Table"
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".

    Returns:
        dict: Dictionary containing document entries formatted for front-end
    """
    check_type(type)
    
    try:
        # Returns list of documents
        uuid_list = []
        for _ , file_name in file_names.items():
            # Split and replace string to get UUID only
            # Example file_name: doc_743c605d-4fa3-4656-aefb-a03dbfef65d8.txt
            document_id = file_name.split(".")[0].replace("doc_", "")
            uuid_list.append(document_id)

        document_ids_list = get_documentids_from_uuid(uuid_list, type=type)

        if len(document_ids_list) < 1:
            logging.warning("No Document IDs found from UUID Document Names. Maybe because there are no results.")
        
        source_documents = get_documents_from_document_ids(document_ids_list, container, type=type)

        if len(source_documents) < 1:
            logging.warning("No Documents found from Document IDs. Maybe because there are no results.")
        
        return source_documents
    
    except Exception as e:
        logging.error(f"Exception:\n{e}")
        print(e)


def get_documentids_from_uuid(uuid_list, type="news"):
    """Get document IDs mapped from UUIDs.

    Args:
        uuid_list (list): List of UUIDs
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".

    Returns:
        list: List of document IDs mapped from their UUIDs
    """
    check_type(type)
    
    try:
        with open(os.environ['MAPPING_DIR'] + f"{type}_mapping.json", 'r') as fp:
            mapping_dict = json.load(fp)

        #return [mapping_dict.get(uid, "None") for uid in uuid_list]
        return [mapping_dict.get(uid) for uid in uuid_list]

    except FileNotFoundError as fnfe:
        logging.error(f"Mapping file not found for {type}.")
        logging.error(f"Exception:\n{fnfe}")
        print(fnfe)
        
    except Exception as e:
        logging.error(f"Exception:\n{e}")
        print(e)

def get_documents_from_document_ids(document_ids_list, table, type="news"):
    """Get document entries from DynamoDB given a list of their IDs and type.

    Args:
        document_ids_list (list): List containing news_ID or tweets_ID values
        table (dynamodb.Table): DynamoDB Table, either "News_Table or "Tweets_Table"
        type (str, optional): Type, either "news" or "tweets". Defaults to "news".

    Returns:
        dict: Dictionary of the form f"content{i}": document_entry, where document_entry is formatted.
    """
    check_type(type)
    documents_dict = {}

    key_month = f"{type}_month"
    key_id = f"{type}_ID"
    ner_flag = f"{type}_ner_flag"
    
    if type == "tweets":
        # Correct to "tweet_ner_flag" from "tweets_ner_flag"
        ner_flag = "tweet_ner_flag"
    
    for i, document_id in enumerate(document_ids_list):
        try:
            # document_id is a list containing [id, datetime]
            # Extract month from the datetime to be used as Key
            document_month = dt.datetime.fromisoformat(document_id[1]).strftime("%Y-%m")
            result = table.query(
                KeyConditionExpression=Key(key_month).eq(document_month) & Key(key_id).eq(document_id[0]), 
                FilterExpression=Attr(ner_flag).eq(True)
            )["Items"]
            
            if result:
                required_docs = transform_to_ui_format(result[0], i, type=type)
                documents_dict[f"content{i}"] = required_docs
        except Exception as e:
            logging.error("Exception getting documents from DynamoDB.")
            logging.error(f"Exception:\n{e}")
            print(e)
            
    return documents_dict


def transform_to_ui_format(item, index, type="news"):
    """Transform a news/tweet entry to UI format. It replaces the news/tweet with an HTML formatted
    string showing sentiment, and returns a dictionary consisting of elements required for front-end.

    Args:
        item (dict): Dictionary containing a news/tweet entry
        index (int): Index of news/tweet in the search prediction
        type (str, optional): Type of entry, either "news" or "tweets". Defaults to "news".

    Returns:
        dict: Dictionary consisting of elements required for front-end.
    """
    check_type(type)
    
    if type == "news":
        text_key = "long_description"
        date_key = "news_date"
    elif type == "tweets":
        text_key = "tweet_text"
        date_key = "tweet_date"
        
    html_str = ""
    
    pos_sent, neg_sent = 0, 0
    sentiment_class = ""
    sentiment_score = 0.0
    if isinstance(item["sentiment_analysis"], list):
        for sentence in item["sentiment_analysis"]:
            if sentence and isinstance(sentence, dict):
                if sentence["sentiment_class"] == "positive":
                    html_str = f"<{sentence['sentiment_class']}>{sentence['text']}</{sentence['sentiment_class']}>"
                    item[text_key] = item[text_key].replace(sentence["text"], html_str)
                    pos_sent += 1
                elif sentence["sentiment_class"] == "negative":
                    html_str = f"<{sentence['sentiment_class']}>{sentence['text']}</{sentence['sentiment_class']}>"
                    item[text_key] = item[text_key].replace(sentence["text"], html_str)
                    neg_sent += 1
                else:
                    html_str = f"<normal>{sentence['text']}</normal>"
                    item[text_key] = item[text_key].replace(sentence["text"], html_str)
        
    if (pos_sent == 0) and (neg_sent == 0):
        sentiment_score = 0.0
    else:
        sentiment_score = (pos_sent - neg_sent) / (pos_sent + neg_sent)
    
    
    if sentiment_score > 0:
        sentiment_class = "positive"
    elif sentiment_score < 0:
        sentiment_class = "negative"
    else:
        sentiment_class = "neutral"
     
    # Title-case type   
    ttype = type.title()
    
    result_dict = {
        f"{ttype}_Title": item.get("title", ''),
        f"{ttype}_Source": item.get(f"{type}_source", ''),
        f"{ttype}_Sentiment": sentiment_class,
        f"{ttype}_Time_Stamp": item[date_key].split()[1],
        f"{ttype}_Sentiment_Score": round(sentiment_score, 2),
        f"{ttype}_Summary": item.get("summary", ''),
        f"{ttype}_Tags": [],
        f"{ttype}_ID": index,
        "ID":item[f"{type}_ID"],
        "Events_Detected": item["All_Events"],
        ttype: item[text_key],
        f"Actual_{ttype}": item[text_key],
        f"{ttype}_date": item[date_key],
        f"Sentiment_Analyzed_{type}": item[text_key],
    }
    return result_dict


def process_qna_prediction(pred, news_pred, news_search_results, news):
    """Process the predictions returned by Reader for Question and Answering.

    Args:
        pred (dict): Prediction dictionary returned by the Reader
        news_pred (dict): News search prediction directly from the Search pipeline
        news_search_results (dict): Simple news search results after processing
        news (dict): Detailed News search results after processing raw predictions from the Search pipeline

    Returns:
        dict, dict: QnA Results, QnA Scores
    """
    
    try:
        answers = pred['answers']
        
        # Get the Haystack Document Store's Document ID for the News Results
        news_doc_ids = [doc.id for doc in news_pred['documents']]

        qna_results = []
        for i in range(len(answers)):
            # Get the ID of the document the i-th answer came from
            ans_doc_id = answers[i].document_id

            # Get the position of the document in the News search results
            news_index = news_doc_ids.index(ans_doc_id)

            # Get the corresponding processed news results
            processed_news_search = news_search_results[f"content{news_index}"]
            processed_news = news[f"content{news_index}"]
            
            qna_dict = {
                "content_name": f"content{news_index}",
                "answer": answers[i].answer,
                "context": answers[i].context,
                "News_Search_Results": {
                    "search_results": processed_news_search,
                    "news": processed_news
                }
            }

            qna_results.append(qna_dict)

        # qna_results = [
        #     {
        #         "content_name": f"content{i}",
        #         "answer": answers[i].answer,
        #         "context": answers[i].context
        #     }
        #     for i in range(len(answers))
        # ]
        
        qna_scores = {
            f"content{i}_score": answers[i].score
            for i in range(len(answers))
        }

        return qna_results, qna_scores
    
    except KeyError as ke:
        logging.error("No QnA answers in prediction.")
        logging.error(f"Exception:\n{ke}")
        print(ke)
        
        return {}, {}
    
    except Exception as e:
        logging.error(f"Exception:\n{e}")
        print(e)
        
        return {}, {}
    


# https://alexwlchan.net/2020/06/finding-the-months-between-two-dates-in-python/
def months_between(start_date, end_date):
    """Given two instances of ``datetime.date``, generate a list of dates on the 
    1st of every month between the two dates (inclusive).

    Args:
        start_date (datetime.date): Start date
        end_date (datetime.date): End date

    Raises:
        ValueError: If the end date is before the start date

    Yields:
        datetime.date: A date containing the 1st of a month in between
    """
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is not before end date {end_date}")

    year = start_date.year
    month = start_date.month

    while (year, month) <= (end_date.year, end_date.month):
        yield dt.date(year, month, 1)

        # Move to the next month.  If we're at the end of the year, wrap around
        # to the start of the next.
        #
        # Example: Nov 2017
        #       -> Dec 2017 (month += 1)
        #       -> Jan 2018 (end of year, month = 1, year += 1)
        #
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1


def get_today():
    """Get today's date as a datetime.date object.

    Returns:
        datetime.date: Today's date
    """
    # UTC Timezone
    UTC = dt.timezone.utc
    
    # Define IST Timezone
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30), name='IST')
    
    # Get current datetime in UTC, make it timezone aware as UTC, then transform into IST
    now_dt = dt.datetime.utcnow().replace(tzinfo=UTC).astimezone(IST)
    
    # Format as YYYY-MM-DD and then use that to create date object
    today = dt.date.fromisoformat(now_dt.strftime("%Y-%m-%d"))
    
    return today


def convert_datetime_to_date(datetime_str):
    """Convert an ISO Format Datetime string to a date object.

    Args:
        datetime_str (str): Datetime string in ISO Format

    Returns:
        datetime.date: A date object extracted from the datetime string
    """
    # Create a datetime object
    item_datetime = dt.datetime.fromisoformat(datetime_str)
    
    # Get the date in YYYY-MM-DD format
    item_date_str = item_datetime.strftime("%Y-%m-%d")
    
    # Create a date object from the date string
    item_date = dt.date.fromisoformat(item_date_str)
    
    return item_date


def outside_threshold(item_date, threshold_date):
    """Check whether the given date is before the threshold date. This function is meant to be used
    for checking whether a given news/tweet is older than a given number of days or not.

    Args:
        item_date (datetime.date): The given date to check
        threshold_date (datetime.date): The threshold date

    Returns:
        bool: True if the given date is before the threshold date.
    """
    return item_date < threshold_date