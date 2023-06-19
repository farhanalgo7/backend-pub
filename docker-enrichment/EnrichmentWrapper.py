""" 
Task of this module:
1. To get data from AWS dynamo DB which is not enriched
2. Perform enrichment of data which is not enriched
3. Update the source table with enriched data

Prerequisites
  !pip install boto3
  !pip install nltk

Author: Ekansh Gupta <egupta@alagoanalytics.com>
AlgoAnalytics
"""


########## importing all dependencies#############

import boto3
from boto3.dynamodb.conditions import Key, Attr
import nltk
import json
import requests as s
import time
import ast
import sys
from decimal import Decimal
import datetime as dt
from datetime import timezone

#nltk.download("punkt")


# ----------------------------- SECTION -Service discovery to get Ip address-------------------------------------------------------------


# using servive discovery resource to get enrichment service IP address/ endpoint
serviceclient = boto3.client("servicediscovery", region_name="ap-south-1")
response = serviceclient.discover_instances(NamespaceName="microservices", ServiceName="microservices")
ip = response["Instances"][0]["Attributes"]["AWS_INSTANCE_IPV4"]

endpoint_url = "http://" + ip + ":5000/items/"


# ----------------------------- SECTION- Connection to DynamoDB-------------------------------------------------
""" About function get_connection
Function to connect to AWS resource (Dynamo DB)
Input : None
Output: return connection object
"""


def get_connection():
    algoDDB = boto3.resource(
        "dynamodb",
        region_name="ap-south-1"
        
    )
    return algoDDB


# initializing dynamodb tables containing News and Tweets
tweets_table = get_connection().Table("Tweets_Table")
news_table = get_connection().Table("News_Table")

# ------------------------------------------------------------------------------------------------------------------


# ----------------------------- SECTION- Functions Declaration------------------------------------------------------


"""  About function enrichment
Function to send batch of data to enrichment API to perform enrichment
Input : batch_of_news - contains list of string (News or Tweets)
        endpoint_url  - contains the endpoint URL of API
Output: dictionary containing sentiment and Ner in format = {'sentiment_analysis': '[..]' , 'ner_analysis' : '[..]}
"""


def enrichment(batch_of_news, endpoint_url):
    # ***************** calling NER API to perform NER and get output in form {'sentiment_analysis' : [], 'ner_analysis':[]}
    response = s.post(endpoint_url, json=batch_of_news).json()
    return response


"""  About function fetch_data
Function to fetch data from AWS dynamoDB for given table
Input : table        - One of the initialized tables from which data needs to be fetched 
        filter_attr  - To fetch only the records meeting a condition based on 'filter_attr' column ..example:select * from table where filter_attr = Null
Output: list dictionary containing dictionary of fetched items
"""


def fetch_data(table, filter_attr):

    this_month = (
        dt.datetime.now(timezone.utc) + dt.timedelta(hours=5, minutes=30)
    ).strftime("%Y-%m")
    today = dt.date.today()
    first = today.replace(day=1)
    lastMonth = first - dt.timedelta(days=1)
    last_month = lastMonth.strftime("%Y-%m")
    list_months = [this_month, last_month]
    print("\n list of months to be scanned ", list_months)

    if table.table_name == "News_Table":
        limit=50
        key = "news_month"
    if table.table_name == "Tweets_Table":
        limit=100
        key = "tweets_month"
    # Scanning the table for items with no NER
    print(
        "*************scanning ",
        table.table_name,
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

    # items= resp_['Items']

    print("************ Fetched all records for ",table.table_name,"*****************************************")
    print("************ Total records fetched : ",len(items),"*********************************************")

    return items


"""  About function perform_enrichment
Function to perform enrichment for given items 
Input : table  - One of the initialized tables from which data needs to be fetched 
        items  - Items of Dynamo DB received from fetch_data fucntion
Output: None
"""


def perform_enrichment(table, items):
    cnt = 0
    for item in items:

        if table.table_name == "Tweets_Table":
            text = item["tweet_text"]
        if table.table_name == "News_Table":
            text = item["title"] + ". " + item["long_description"]

        # print("\n text is :", nltk.tokenize.sent_tokenize(text))
        # making a batch for all sentences in a particular news
        try:
            batch = nltk.tokenize.sent_tokenize(text)
        except:
            continue

        try:
            # calling encrichment fucntion to perform enrichment
            enriched_data = enrichment({"article": batch}, endpoint_url)
            print(enriched_data)
        except:
            # try for one more time
            try:
                print(
                    "Exception while enriching ",
                    text[:50],
                    "...trying for one more time...........",
                )
                enriched_data = enrichment({"article": batch}, endpoint_url)
                time.sleep(5)
            # assign null for NER and Sentiment for that particular news article
            except Exception as err:
                enriched_data = {"Ner": "Null", "Sentiment": "Null", "events": "Null"}
                print(
                    "could not perform enrichment for text starting with:",
                    text[:100],
                    " getting error :",
                    err,
                )
                Update_table(table, item, "Null", "Null", "Null", "Null")
                continue
        print("\n\n", enriched_data)
        Sentiment = ast.literal_eval(enriched_data["sentiment_analysis"])
        Ner = ast.literal_eval(enriched_data["ner_analysis"])
        Events = ast.literal_eval(enriched_data["events"])
        formatted_sentiment = json.loads(json.dumps(Sentiment), parse_float=Decimal)
        formatted_ner = json.loads(json.dumps(Ner), parse_float=Decimal)
        formatted_events = json.loads(json.dumps(Events), parse_float=Decimal)
        print(enriched_data["summary"])
        if table.table_name == "News_Table":
            Summary = enriched_data["summary"]
        else:
            Summary = []

        if (len(Sentiment) == 0) and len(Ner) == 0:
            print(
                "\nNo enriched data received for text starting with ",
                text[:100],
                "..............",
            )

        else:
            cnt += 1

        # ************ updating source table **************
        Update_table(
            table, item, formatted_sentiment, formatted_ner, Summary, formatted_events
        )

    print(
        "*****************",
        table.table_name,
        " table updated with enriched data******************************",
    )
    print("*****************Total records Updated: ", cnt)
    print("*****************records with no enriched data: ", (len(items) - cnt))


"""  About function Update_table
Function to perform enrichment for given items 
Input : table  - One of the initialized tables from which data needs to be fetched 
        item  - item (record) which need to be updated
        Sentiment- Sentiment list to be updated to item
        Ner- Ner list to be updated to item
Output: None
"""


def Update_table(table, item, Sentiment, Ner, summary, events):
    event_flag = False
    if len(events) == 0:
        event_flag = False
    else:
        event_flag = True
    if table.table_name == "Tweets_Table":
        table.update_item(
            Key={"tweets_month": item["tweets_month"], "tweets_ID": item["tweets_ID"]},
            UpdateExpression="set tweet_ner_flag = :flag , ner_analysis= :ner , sentiment_analysis = :sentiment, event_analysis = :events ,events_flag = :events_flag ",
            ExpressionAttributeValues={
                ":flag": True,
                ":ner": Ner,
                ":sentiment": Sentiment,
                ":events": events,
                ":events_flag": event_flag,
            },
        )
    if table.table_name == "News_Table":
        table.update_item(
            Key={"news_month": item["news_month"], "news_ID": item["news_ID"]},
            UpdateExpression="set news_ner_flag = :flag , ner_analysis= :ner , sentiment_analysis = :sentiment , summary = :summary, event_analysis = :events,events_flag = :events_flag",
            ExpressionAttributeValues={
                ":flag": True,
                ":ner": Ner,
                ":sentiment": Sentiment,
                ":summary": summary,
                ":events": events,
                ":events_flag": event_flag,
            },
        )

    print("\n........ One record Updated....... \n")


# ______________________________________________________________________________________________________________________________

# ******************************************** TWEETS TABLE starting point ***********************************************
while 1:
    # fetch items from tweets_table
    tweet_items = fetch_data(tweets_table, "tweet_ner_flag")
    # if no items are found with 'tweet_ner_flag'=False then break the code since all items are updated with enriched data
    if len(tweet_items) == 0:
        break
    else:
        perform_enrichment(tweets_table, tweet_items)

# ******************************************** NEWS TABLE starting point ***********************************************
while 1:
    # fetch items from news_table
    news_items = fetch_data(news_table, "news_ner_flag")
    # if no items are found with  'ner' =False then break the code since all items are updated with enriched data
    if len(news_items) == 0:
        break
    else:
        perform_enrichment(news_table, news_items)

