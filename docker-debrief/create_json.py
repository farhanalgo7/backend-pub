from helper import write_json_file
from get_dynamodb_data import(
    sum_get_total_news,
    read_data_dynamodb,
    
    sum_get_total_tweets

)
from access_file import get_secret
# import config
import boto3
import time
import mysql.connector
import os
import json
import numpy as np
from datetime import datetime, timedelta, date
from collections import defaultdict, OrderedDict
from dateutil.relativedelta import relativedelta, FR
import calendar
import datetime as dt
import simplejson as json
from pytz import timezone
import exchange_calendars as tc
xbom = tc.get_calendar("XBOM")
india_time = timezone("Asia/Kolkata")
import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
import pandas_market_calendars as mcal




# Custom dictionary data structure function
def dict_None_output():
    return None


def dict_output():
    return defaultdict(dict_None_output)


def updated_dict():
    return defaultdict(float)


# Function to check whether there is holiday or not
logging.info('get secret')


def working_check():
    start_date = date(date.today().year, 1, 1)
    end_date = start_date + dt.timedelta(days=400)
    bse = mcal.get_calendar('BSE')
    early = bse.schedule(start_date=str(start_date), end_date=str(end_date))
    early.reset_index(drop=False, inplace=True)
    list_working_days = []
    for i in early["index"]:
        list_working_days.append(str(i.date()))
    return list_working_days


list_working_days = working_check()
logging.info('working_check')

# storing the secret manager data and dividing them into different variables

# Check if data debrief is running between 9:10 AM to 5:15 PM
nw = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)
hrs = nw.hour
mins = nw.minute
secs = nw.second
zero = dt.timedelta(seconds=secs+mins*60+hrs*3600)
st = nw - zero
time1 = st + dt.timedelta(seconds=9*3600+10*60)   # this gives 09:10 AM
time2 = st + dt.timedelta(seconds=17*3600+15*60)  # this gives 05:15 PM
#nw = dt.datetime(2022, 3, 31, 9, 20, 48, 140961, tzinfo=dt.timezone.utc)
print("create_json.py:", nw, time1, time2)


# Fetch secret keys

credentials = json.loads(get_secret()["SecretString"])



host = credentials["host"]
user = credentials["username"]
port = credentials["port"]
database = credentials["Database_Name"]
region = credentials["region_name"]
rds_stock_data_table = credentials["Ingestion_table_name"]
rds_stock_next5Days_prediction_table = credentials[
    "rds_stock_next5Days_prediction_table"
]
rds_stock_last5Days_prediction_table = credentials[
    "rds_stock_last5Days_prediction_table"
]
rds_stock_prediction_table = credentials["rds_stock_prediction_table"]
trends_prediction_stock_table = credentials["trends_prediction_stock_table"]

default_bucket = credentials["default_bucket"]
pdf_topic_bucket = credentials["pdf-topic-bucket"]
topic_bucket = credentials["topic_bucket"]

# default_bucket = "algoanalytics-fabric-website"
# pdf_topic_bucket = "algoanalytics-fabric-website"
# topic_bucket = "topics-prediction-bucket"


logging.warning('credentials')
""" Connection to s3  """
s3_client = boto3.client('s3',region_name="ap-south-1")
# aws_access_key_id=config.algo_access_key, aws_secret_access_key=config.algo_secret_access_token)

s3 = boto3.resource("s3", region_name="ap-south-1")
# ,aws_access_key_id=config.algo_access_key, aws_secret_access_key=config.algo_secret_access_token)

s3_topic_bucket = s3.Bucket(topic_bucket)
client = boto3.client("rds", region_name="ap-south-1")
# ,aws_access_key_id=config.algo_access_key, aws_secret_access_key=config.algo_secret_access_token)

os.environ["LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN"] = "1"
host = "fabric-database-1.cptxszv3ahgy.ap-south-1.rds.amazonaws.com"
region = "ap-south-1"
port = 3306
logging.warning('Connection to s3')



def read_data_from_s3(bucket,filename):
    content_object = s3.Object(bucket, "Json/{}".format(filename))
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.loads(file_content)
    logging.warning('read_data_from_s3')
    return json_content


""" Connection to rds instance  """

token = client.generate_db_auth_token(
    DBHostname=host, Port="3306", DBUsername="fabric_user", Region=region
)
if (nw >= time1 and nw <= time2) and (str(nw.date()) in list_working_days):
# con=mysql.connector.connect(host=host, user="fabric_user", password=token)
    con = mysql.connector.connect(
        host=host, user=user, password=token, port=port, database=database,buffered=True
    )
    cursor = con.cursor()
    cursor.execute("use StockData")
    logging.info('Connection to rds instance')


# print("\nListing blobs...")

# # List the blobs in the container
# blob_list = container_client.list_blobs()
# for blob in blob_list:
#     print("\t" + blob.name)

# fetch Summary Data
def fetch_data_summary(
    ticker_name,
    today_date,
    summary_data_days,
    summary_json,
    ticker_id,
    all_tweets,
    all_news,
):
    """
    Create Summary dict for a given ticker from since to until

    Args:
        ticker_name (string): Ticker name
        today_date(datetime): Current date in IST
        summary_data_days(datetime): date to which have to generate summary
        summary_json(dict): Summary with updated date
        ticker_id(string): ticker Symbol
        all_tweets(dict): all tweets groupby ticker_id
        all_news(dict): all news groupby ticker_id



    Returns:
        dict: Dictionary of today news with colors as per percentile.
    """

    # Getting news sentiment and total news for todaya and 30 days 
    (
        news_count,
        # required_news,
        # total_news,
        # total_news_sentences_sentiment,
        # total_news_sentences_pos_sentiment,
        # total_news_sentences_neg_sentiment,
        # total_news_sentences_neutral_sentiment,
        news_sent,
        today_news,
        today_news_sent,
    ) = sum_get_total_news(
        ticker_name,
        summary_data_days,
        today_date,
        all_news,
    )
    print("\n\nTODAY NEWS:\n\n", today_news)
    # Getting tweets sentiment and total tweets for todaya and 30 days
    (
        tweet_count,
        # required_tweets,
        # total_tweets,
        # total_tweet_sentences_sentiment,
        # total_tweet_sentences_pos_sentiment,
        # total_tweet_sentences_neg_sentiment,
        # total_tweet_sentences_neutral_sentiment,
        tweets_sent,
        today_tweets,
        today_tweet_sent,
    ) = sum_get_total_tweets(
        ticker_name,
        summary_data_days,
        today_date,
        all_tweets,
    )
    print("\n\nTODAY TWEETS: ", today_tweets, "\n\n")
    news_quartile = []
    print("\n\nNEWSCOUNT = ", news_count, "\n\n")
    print("\n\nNEWS COUNT LENGTH = ", len(news_count), "\n\n")
    for item in news_count:
        # print("\n\nITEM['NEWSCOUNT'] = ", item["newscount"], "\n\n")
        news_quartile.append(item["newscount"])
    tweet_quartile = []
    print("\n\nTWEET COUNT = ", tweet_count, "\n\n")
    print("\n\nTWEET COUNT LENGTH = ", len(tweet_count), "\n\n")
    for item in tweet_count:
        # print("\n\nITEM['TWEETCOUNT'] = ", item["tweetcount"], "\n\n")
        tweet_quartile.append(item["tweetcount"])

    # getting stock price data
    today_date = today_date.date()
    summary_data_days = summary_data_days.date()
    print("\n\nTODAY DATE: ", today_date, "\n\n")
    print("\n\nTODAY DATE TYPE: ", type(today_date), "\n\n")
    if (nw >= time1 and nw <= time2) and (str(nw.date()) in list_working_days):     
    # if(True):
        print("\n\nQUERY:\n\nselect Open, Close, Volume, DATE_FORMAT(date, '%Y-%m-%d')  from {} where ticker_id='{}' and DATE_FORMAT(date, '%Y-%m-%d') between '{}' and '{}'  \n\n".format(
        rds_stock_data_table, ticker_id, summary_data_days, today_date
        ))
        
        cursor.execute("select Open, Close, Volume, date(date)  from {} where ticker_id='{}' and date(date) between '{}' and '{}'  ".format(
            rds_stock_data_table, ticker_id, summary_data_days, today_date
            )
        )
        # generating total news, tweets volume, returns, sentiment
        fetched_result = {"Records": list(map(list, cursor.fetchall()))}
        # print(fetched_result)
        filename1 = "{}/fetched_result.json".format(ticker_id)
        print("Writing to s3 for {}".format(ticker_id))
        write_json_file(filename1, fetched_result)
        fetched_result = fetched_result["Records"]
        volumes = []
        returns = []
        total_volumes = 0
        total_retuns = fetched_result[-1][1] - fetched_result[0][0]
        offset = max(1, (today_date.weekday() + 6) % 7 - 5)
        time_yestarday = timedelta(offset)
        most_recent = today_date - time_yestarday
        print("today_date.weekday()",today_date)
        print("ttime_yestarday",time_yestarday)
        print("most_recent",most_recent)
        print("offset",offset)
        while most_recent.strftime('%Y-%m-%d') not in list_working_days:
            offset += 1
            time_yestarday = dt.timedelta(offset)
            most_recent = today_date - time_yestarday
            print("today_date: ", today_date)
        # cursor.execute(
        #     "select Close  from {} where ticker_id='{}' and date(date) = '{}' and DATE_FORMAT(date, '%H') = '{}' order by date desc limit 1 ".format(
        #         azure_sql_stock_data_table, ticker_id, most_recent, '15'
        #     )s
        # )
        var="select close  from {}  where DATE_FORMAT(date, '%Y-%m-%d') ='{}' and ticker_id='{}' and DATE_FORMAT(date, '%H') >= '{}' order by date desc limit 1  ".format(
                rds_stock_data_table,most_recent,ticker_id,'15'
            )
        print(var)
        cursor.execute(
        "select close  from {}  where DATE_FORMAT(date, '%Y-%m-%d') ='{}' and ticker_id='{}' and DATE_FORMAT(date, '%H') >= '{}' order by date desc limit 1  ".format(
                rds_stock_data_table,most_recent,ticker_id,'15'
        
        )
        )
        # print("select Close  from {} where ticker_id='{}' and date(date) = '{}' and DATE_FORMAT(date, '%H') = '{}' order by date desc limit 1 ".format(
        #         azure_sql_stock_data_table, ticker_id, most_recent, '15'
        #     ))
        res=list(map(list, cursor.fetchall()))
        print(res)
        
        # yestarday_close = {"Records": res[0][0]}
        filename2 = "{}/yestarday_close.json".format(ticker_id)
        yestarday_close = {}
        if len(res) != 0:
            yestarday_close = {"Records": res[0][0]}
            write_json_file(filename2, yestarday_close)
        else:
            yestarday_close = read_data_from_s3(default_bucket, filename2)

        yestarday_close = yestarday_close["Records"]
        var="select Close  from {} where ticker_id='{}' and DATE_FORMAT(date, '%Y-%m-%d') = '{}' order by date desc limit 1 ".format(
                 rds_stock_data_table, ticker_id, today_date
            )
        print(var)
        cursor.execute(
        "select Close  from {} where ticker_id='{}' and date(date) = '{}' order by date desc limit 1 ".format(
            rds_stock_data_table, ticker_id, today_date
        )
    )
        res2=list(map(list, cursor.fetchall()))
        print("res2",res2)

        filename3 = "{}/today_current.json".format(ticker_id)
        today_current = {}
        if len(res) != 0:
            today_current = {"Records": res2[0][0]}
            write_json_file(filename3, today_current)
        else:
            today_current = read_data_from_s3(default_bucket, filename3)

        today_current = today_current["Records"]
        if yestarday_close < 0:
            today_return = "NA"
        else:
            today_return = ((today_current - yestarday_close)/yestarday_close)*100
        today_volume = int(fetched_result[-1][2])
        print("\n\nFETCHED RESULT: ", fetched_result, "\n\n")
        for item in fetched_result:
            volumes.append(int(item[2]))
            total_volumes += int(item[2])
        for i in range(1,len(fetched_result)):
            returns.append(((fetched_result[i][1] - fetched_result[i-1][1])/fetched_result[i-1][1])*100)

        tweets_sent = dict(tweets_sent)
        total_tweets_sent = []
        for item in tweets_sent.values():
            if item["total"]:
                total_tweets_sent.append(item["sentiment_score"] / item["total"])
            else:
                total_tweets_sent.append(0.0)

        news_sent = dict(news_sent)
        total_news_sent = []
        for item in news_sent.values():
            if item["total"]:
                total_news_sent.append(item["sentiment_score"] / item["total"])
            else:
                total_news_sent.append(0.0)
        
        # Generating Quartiles
        tweets_sent_quantile = list(np.percentile(total_tweets_sent, [25, 50, 75, 100]))
        news_sent_quantile = list(np.percentile(total_news_sent, [25, 50, 75, 100]))
        v_quantile = list(np.percentile(volumes, [25, 50, 75, 100]))
        r_quantile = list(np.percentile(returns, [25, 50, 75, 100]))
        news_quartile = list(np.percentile(news_quartile, [25, 50, 75, 100]))
        tweet_quartile = list(np.percentile(tweet_quartile, [25, 50, 75, 100]))
        if (yestarday_close < 0):
            result_summary = {
                "Today_News_Volume": today_news,
                "Today_News_Volume_percentile": news_quartile,
                "Today_Tweet_Volume": today_tweets,
                "Today_Tweet_Volume_percentile": tweet_quartile,
                "Today_Returns_percentile": r_quantile,
                "Today_Volume": today_volume,
                "Today_Volume_percentile": v_quantile,
                "Today_Returns":  today_return,
                "Today_News_Sentiment":  round(today_news_sent, 2),
                "Today_News_Sentiment_percentile": news_sent_quantile,
                "Today_Tweet_Sentiment":  round(today_tweet_sent, 2),
                "Today_Tweet_Sentiment_percentile": tweets_sent_quantile
            }
        else:
            result_summary = {
                "Today_News_Volume": today_news,
                "Today_News_Volume_percentile": news_quartile,
                "Today_Tweet_Volume": today_tweets,
                "Today_Tweet_Volume_percentile": tweet_quartile,
                "Today_Returns_percentile": r_quantile,
                "Today_Volume": today_volume,
                "Today_Volume_percentile": v_quantile,
                "Today_Returns":  round(float(today_return), 2),
                "Today_News_Sentiment":  round(today_news_sent, 2),
                "Today_News_Sentiment_percentile": news_sent_quantile,
                "Today_Tweet_Sentiment":  round(today_tweet_sent, 2),
                "Today_Tweet_Sentiment_percentile": tweets_sent_quantile
            }
        print("\n\nNEWS VOL PERCENTILE: ", result_summary['Today_News_Volume_percentile'], "\n\n")
        print("\n\nTWEETS VOL PERCENTILE: ", result_summary['Today_Tweet_Volume_percentile'], "\n\n")
        colors = defaultdict(str)
        for key, value in result_summary.items():
            if '_percentile' not in key:
                if "Sentiment" in key and value == 0:
                    if "News" in key and result_summary["Today_News_Volume"] == 0:
                        result_summary[key] = "NA"
                        colors[key+"_colors"] = "#ccc"
                        continue
                    elif "Tweet" in key and result_summary["Today_Tweet_Volume"] == 0:
                        result_summary[key] = "NA"
                        colors[key+"_colors"] = "#ccc"
                        continue
                if value != "NA":
                    if value <= result_summary[key+'_percentile'][0]:
                        # colors[key+"_colors"] = "#84FED2"
                        colors[key+"_colors"] = "#FFC7D4"
                    elif value <= result_summary[key+'_percentile'][2]:
                        colors[key+"_colors"] = "#FFE7A8"
                    else:
                        # colors[key+"_colors"] = "#FFC7D4"
                        colors[key+"_colors"] = "#84FED2"
                if value == "NA":
                    colors[key+"_colors"] = "#84FED2"
        # print(colors)
        # Creating Summary dict
        
        if (yestarday_close < 0):
            summary_json = {
                "Stock_Ticker": ticker_id,
                "Today_News_Volume": today_news,
                "Today_Tweet_Volume": today_tweets,
                "Total_Returns": total_retuns,
                "Today_Volume": today_volume,
                "Today_Returns": today_return,
                "Today_News_Sentiment": round(result_summary["Today_News_Sentiment"], 2) if isinstance(result_summary["Today_News_Sentiment"], float) else result_summary["Today_News_Sentiment"],
                "Today_Tweet_Sentiment": round(result_summary["Today_Tweet_Sentiment"], 2) if isinstance(result_summary["Today_Tweet_Sentiment"], float) else result_summary["Today_Tweet_Sentiment"],
                "yestarday_close": yestarday_close,
            }
            summary_json.update(colors)
            return summary_json
        else:
            summary_json = {
                "Stock_Ticker": ticker_id,
                "Today_News_Volume": today_news,
                "Today_Tweet_Volume": today_tweets,
                "Total_Returns": total_retuns,
                "Today_Volume": today_volume,
                "Today_Returns": round(float(today_return), 2),
                "Today_News_Sentiment": round(result_summary["Today_News_Sentiment"], 2) if isinstance(result_summary["Today_News_Sentiment"], float) else result_summary["Today_News_Sentiment"],
                "Today_Tweet_Sentiment": round(result_summary["Today_Tweet_Sentiment"], 2) if isinstance(result_summary["Today_Tweet_Sentiment"], float) else result_summary["Today_Tweet_Sentiment"],
                "yestarday_close": yestarday_close,
            }
            summary_json.update(colors)
            logging.info('fetch_data_summary')
            return summary_json
    else:
        print("Reading Data from s3 for {}".format(ticker_id))
        filename1 = "{}/fetched_result.json".format(ticker_id)
        fetched_result = read_data_from_s3(default_bucket,filename1)
        fetched_result = fetched_result["Records"]
        # print(fetched_result[0])
        volumes = []
        returns = []
        total_volumes = 0
        total_retuns = fetched_result[-1][1] - fetched_result[0][0]
        offset = max(1, (today_date.weekday() + 6) % 7 - 3)
        time_yestarday = dt.timedelta(offset)
        most_recent = today_date - time_yestarday
        filename2 = "{}/yestarday_close.json".format(ticker_id)
        yestarday_close = read_data_from_s3(default_bucket,filename2)
        yestarday_close = yestarday_close["Records"]
        filename3 = "{}/today_current.json".format(ticker_id)
        today_current = read_data_from_s3(default_bucket,filename3)
        today_current = today_current["Records"]
        if (yestarday_close < 0):
            today_return = "NA"
        else:
            today_return = ((today_current - yestarday_close) /
                            yestarday_close)*100
        today_volume = int(fetched_result[-1][2])
        for item in fetched_result:
            volumes.append(int(item[2]))
            total_volumes += int(item[2])
        for i in range(1, len(fetched_result)):
            returns.append(
                ((fetched_result[i][1] - fetched_result[i-1][1])/fetched_result[i-1][1])*100)

        tweets_sent = dict(tweets_sent)
        total_tweets_sent = []
        for item in tweets_sent.values():
            if item["total"]:
                total_tweets_sent.append(
                    item["sentiment_score"] / item["total"])
            else:
                total_tweets_sent.append(0.0)

        news_sent = dict(news_sent)
        total_news_sent = []
        for item in news_sent.values():
            if item["total"]:
                total_news_sent.append(item["sentiment_score"] / item["total"])
            else:
                total_news_sent.append(0.0)

        # Generating Quartiles
        tweets_sent_quantile = list(np.percentile(
            total_tweets_sent, [25, 50, 75, 100]))
        news_sent_quantile = list(np.percentile(
            total_news_sent, [25, 50, 75, 100]))
        v_quantile = list(np.percentile(volumes, [25, 50, 75, 100]))
        r_quantile = list(np.percentile(returns, [25, 50, 75, 100]))
        news_quartile = list(np.percentile(news_quartile, [25, 50, 75, 100]))
        tweet_quartile = list(np.percentile(tweet_quartile, [25, 50, 75, 100]))
        if (yestarday_close < 0):
            result_summary = {
                "Today_News_Volume": today_news,
                "Today_News_Volume_percentile": news_quartile,
                "Today_Tweet_Volume": today_tweets,
                "Today_Tweet_Volume_percentile": tweet_quartile,
                "Today_Returns_percentile": r_quantile,
                "Today_Volume": today_volume,
                "Today_Volume_percentile": v_quantile,
                "Today_Returns":  today_return,
                "Today_News_Sentiment":  round(today_news_sent, 2),
                "Today_News_Sentiment_percentile": news_sent_quantile,
                "Today_Tweet_Sentiment":  round(today_tweet_sent, 2),
                "Today_Tweet_Sentiment_percentile": tweets_sent_quantile
            }
        else:
            result_summary = {
                "Today_News_Volume": today_news,
                "Today_News_Volume_percentile": news_quartile,
                "Today_Tweet_Volume": today_tweets,
                "Today_Tweet_Volume_percentile": tweet_quartile,
                "Today_Returns_percentile": r_quantile,
                "Today_Volume": today_volume,
                "Today_Volume_percentile": v_quantile,
                "Today_Returns":  round(float(today_return), 2),
                "Today_News_Sentiment":  round(today_news_sent, 2),
                "Today_News_Sentiment_percentile": news_sent_quantile,
                "Today_Tweet_Sentiment":  round(today_tweet_sent, 2),
                "Today_Tweet_Sentiment_percentile": tweets_sent_quantile
            }
        print("\n\nNEWS VOL PERCENTILE: ", result_summary['Today_News_Volume_percentile'], "\n\n")
        print("\n\nTWEETS VOL PERCENTILE: ", result_summary['Today_Tweet_Volume_percentile'], "\n\n")
        colors = defaultdict(str)
        for key, value in result_summary.items():
            if '_percentile' not in key:
                if "Sentiment" in key and value == 0:
                    if "News" in key and result_summary["Today_News_Volume"] == 0:
                        result_summary[key] = "NA"
                        colors[key+"_colors"] = "#ccc"
                        continue
                    elif "Tweet" in key and result_summary["Today_Tweet_Volume"] == 0:
                        result_summary[key] = "NA"
                        colors[key+"_colors"] = "#ccc"
                        continue
                if value != "NA":
                    if value <= result_summary[key+'_percentile'][0]:
                        # colors[key+"_colors"] = "#84FED2"
                        colors[key+"_colors"] = "#FFC7D4"
                    elif value <= result_summary[key+'_percentile'][2]:
                        colors[key+"_colors"] = "#FFE7A8"
                    else:
                        # colors[key+"_colors"] = "#FFC7D4"
                        colors[key+"_colors"] = "#84FED2"
                if value == "NA":
                    colors[key+"_colors"] = "#84FED2"
        # print(colors)
        # Creating Summary dict
        if (yestarday_close < 0):
            summary_json = {
                "Stock_Ticker": ticker_id,
                "Today_News_Volume": today_news,
                "Today_Tweet_Volume": today_tweets,
                "Total_Returns": total_retuns,
                "Today_Volume": today_volume,
                "Today_Returns": today_return,
                "Today_News_Sentiment": round(result_summary["Today_News_Sentiment"], 2) if isinstance(result_summary["Today_News_Sentiment"], float) else result_summary["Today_News_Sentiment"],
                "Today_Tweet_Sentiment": round(result_summary["Today_Tweet_Sentiment"], 2) if isinstance(result_summary["Today_Tweet_Sentiment"], float) else result_summary["Today_Tweet_Sentiment"],
                "yestarday_close": yestarday_close,
            }
            summary_json.update(colors)
            return summary_json
        else:
            summary_json = {
                "Stock_Ticker": ticker_id,
                "Today_News_Volume": today_news,
                "Today_Tweet_Volume": today_tweets,
                "Total_Returns": total_retuns,
                "Today_Volume": today_volume,
                "Today_Returns": round(float(today_return), 2),
                "Today_News_Sentiment": round(result_summary["Today_News_Sentiment"], 2) if isinstance(result_summary["Today_News_Sentiment"], float) else result_summary["Today_News_Sentiment"],
                "Today_Tweet_Sentiment": round(result_summary["Today_Tweet_Sentiment"], 2) if isinstance(result_summary["Today_Tweet_Sentiment"], float) else result_summary["Today_Tweet_Sentiment"],
                "yestarday_close": yestarday_close,
            }
            summary_json.update(colors)
            return summary_json


def create_overview(
    ticker_data, list_months, today_date, index, summary_data_days, all_news,all_tweets
):
    try:
        if (nw >= time1 and nw <= time2) and (str(nw.date()) in list_working_days):
        # if(True):
        # Fetching Volume
            cursor.execute(
                "select Volume, DATE_FORMAT(date, '%Y-%m-%d')  from {} where ticker_id='{}' and DATE_FORMAT(date, '%Y-%m-%d') between '{}' and '{}' and DATE_FORMAT(date, '%H') >= '{}'".format(
                    rds_stock_data_table,
                    ticker_data["Stock_Ticker_Symbol"],
                    summary_data_days,
                    today_date,
                    "15")
                )
            
            yesterday = {"Records": list(map(list, cursor.fetchall()))}
            filename4 = "{}/yesterday1.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename4, yesterday)
            yesterday = yesterday["Records"]
            news_count = defaultdict(int)
            tweet_count = defaultdict(int)
            news_data = []

            # Generating Events
            for item in all_news[ticker_data["Ticker_Name"]]:
                news_count[item["news_date"].split()[0]] += 1
                if (
                    item["news_date"].split()[0]
                    == (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
                        and "Top_Events" in item.keys()
                ):
                    # for i in item["event_analysis"]:
                    #     news_data.append(
                    #         {
                    #             "Event_Type": "News",
                    #             "Event_Name": i,
                    #             "Event_URL": item["link"],
                    #             "Event_ID": item["news_ID"],
                    #         }
                    #     )
                    if item["Top_Events"] != "Null":
                        for i in item["Top_Events"]:
                                # print("All Events i : ", i)
                                # print("Type of All_Events i : ", type(i))
                                news_data.append(
                                    {
                                        "Event_Type": "News",
                                        # "Event_Name": str(i["text"] + "[" + i["labels"]+ "]"),
                                        "Event_Name": i['text'],
                                        "Events_Detected": str(i['labels']),
                                        "Event_URL": item["link"],
                                        "Event_ID": item["news_ID"],
                                    }
                                )
            for item in all_tweets[ticker_data["Ticker_Name"]]:
                    tweet_count[item["tweet_date"].split()[0]] += 1
                    if (
                        item["tweet_date"].split()[0]
                        == (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
                        and "Top_Events" in item.keys()
                    ):
                        if item["Top_Events"] != "Null":
                            for i in item["Top_Events"]:
                                # print("All Events i : ", i)
                                # print("Type of All_Events i : ", type(i))
                                news_data.append(
                                    {
                                        "Event_Type": "Tweets",
                                        # "Event_Name": str(i["text"] + "[" + i["labels"]+ "]"),
                                        "Event_Name": i['text'],
                                        "Events_Detected": str(i['labels']),
                                        "Event_URL": item["tweet_link"],
                                        "Event_ID": item["tweets_ID"],
                                    }
                                )
            # News vs volume data creation
            news = []
            vol_data = []
            date_data = []
            tweets = []

            for vol, date in yesterday:
                    news.append(news_count[date])
                    tweets.append(tweet_count[date])
                    vol_data.append(int(vol))
                    date_data.append(dt.datetime.strptime(
                        date, "%Y-%m-%d").strftime("%d %b %Y"))
                # total_news, dates = get_total_news()
            graph = {"Volume": vol_data, "News": news,
                        "Tweets": tweets, "date": date_data}
            logging.info('create overview')
            return {
                "Graph": graph,
                "Events": news_data,
            }
        
        else:
            filename4 = "{}/yesterday1.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            yesterday = read_data_from_s3(default_bucket,filename4)
            yesterday = yesterday["Records"]
            news_count = defaultdict(int)
            tweet_count = defaultdict(int)
            news_data = []

            # Generating Events
            for item in all_news[ticker_data["Ticker_Name"]]:
                news_count[item["news_date"].split()[0]] += 1
                if (
                    item["news_date"].split()[0]
                    == (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
                    and "All_Events" in item.keys()
                ):
                    # for i in item["All_Events"]:
                    #     percentile = 0.8
                    #     threshold = percentile * float(i['probability'][0])
                    #     probability_list = [float(probability) for probability in i['probability'] if float(
                    #         probability) >= threshold]
                    #     probability_list = probability_list[0:3]
                    #     total_top_labels = len(probability_list)
                    #     top_labels_dict = {
                    #         'text': i['text'], 'labels': i['labels'][:total_top_labels], 'probability': probability_list}
                    #     news_data.append(
                    #         {
                    #             "Event_Type": "News",
                    #             "Event_Name": top_labels_dict['text'],
                    #             "Events_Detected": str(top_labels_dict['labels']),
                    #             "Event_URL": item["link"],
                    #             "Event_ID": item["news_ID"],
                    #         }
                    #     )
                    if item["Top_Events"] != "Null":
                        for i in item["Top_Events"]:
                            #print("All Events i : ", i)
                            # print("Type of All_Events i : ", type(i))
                            news_data.append(
                                {
                                    "Event_Type": "News",
                                    # "Event_Name": str(i["text"] + "[" + i["labels"]+ "]"),
                                    "Event_Name": i['text'],
                                    "Events_Detected": str(i['labels']),
                                    "Event_URL": item["link"],
                                    "Event_ID": item["news_ID"],
                                }
                            )
            for item in all_tweets[ticker_data["Ticker_Name"]]:
                tweet_count[item["tweet_date"].split()[0]] += 1
                if (
                    item["tweet_date"].split()[0]
                    == (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
                    and "All_Events" in item.keys()
                ):
                    # for i in item["All_Events"]:
                    #     percentile = 0.8
                    #     threshold = percentile * float(i['probability'][0])
                    #     probability_list = [float(probability) for probability in i['probability'] if float(
                    #         probability) >= threshold]
                    #     probability_list = probability_list[0:3]
                    #     total_top_labels = len(probability_list)
                    #     top_labels_dict = {
                    #         'text': i['text'], 'labels': i['labels'][:total_top_labels], 'probability': probability_list}
                    #     news_data.append(
                    #         {
                    #             "Event_Type": "Tweets",
                    #             # "Event_Name": str(i["text"] + "[" + i["labels"]+ "]"),
                    #             "Event_Name": top_labels_dict['text'],
                    #             "Events_Detected": str(top_labels_dict['labels']),
                    #             "Event_URL": item["tweet_link"],
                    #             "Event_ID": item["tweets_ID"],
                    #         }
                    #     )
                    if item["Top_Events"] != "Null":
                        for i in item["Top_Events"]:
                            #print("All Events i : ", i)
                            # print("Type of All_Events i : ", type(i))
                            news_data.append(
                                {
                                    "Event_Type": "Tweets",
                                    # "Event_Name": str(i["text"] + "[" + i["labels"]+ "]"),
                                    "Event_Name": i['text'],
                                    "Events_Detected": str(i['labels']),
                                    "Event_URL": item["tweet_link"],
                                    "Event_ID": item["tweets_ID"],
                                }
                            )
            # News vs volume data creation
            news = []
            vol_data = []
            date_data = []
            tweets = []

            for vol, date in yesterday:
                news.append(news_count[date])
                tweets.append(tweet_count[date])
                vol_data.append(int(vol))
                date_data.append(dt.datetime.strptime(
                    date, "%Y-%m-%d").strftime("%d %b %Y"))
            # total_news, dates = get_total_news()
            graph = {"Volume": vol_data, "News": news,
                    "Tweets": tweets, "date": date_data}
            return {
                "Graph": graph,
                "Events": news_data,
            }
    except Exception as e:
        print("Exception in overview ", e)


def create_tweets(ticker_data, all_tweets):
    try:
    
        data_collect = []
        events_json = {}
        # container_client.list_blobs()
        for obj in s3_topic_bucket.objects.all():
                key = obj.key
                if key == "Json/Tweets/{}".format(ticker_data + ".json"):
                    events_json = json.loads(obj.get()["Body"].read().decode('utf-8'))
            #print("event_json",events_json)
        word_cloud_freq = {}
        for keys, values in events_json.items():
            word_cloud_freq[keys] = []
            if isinstance(values, str):
                word_cloud_freq[keys] = [
                    {"id": 1, "text": "DATA NOT AVAILABLE", "ids":[]}
                ]
            else:
                cnt = 1
                for key, item in values["Topic_Tweets_ID"].items():
                    word_cloud_freq[keys].append({"id": cnt, "text": key,  "ids":item})

        for index, item in enumerate(all_tweets[ticker_data]):
            str1 = ""
            item['updated_tweet'] =  item["tweet_text"]
            if (
                item["tweet_ner_flag"]
                and "sentiment_analysis" in item.keys()
            ):
                pos_sent, neg_sent = 0, 0
                sentiment_class = ""
                sentiment_score = 0.0
                try: 
                # print(item["sentiment_analysis"])
                # print("TYPE--->",type(item["sentiment_analysis"]))
                # print(str(item["sentiment_analysis"]))
                    item["sentiment_analysis"] = json.loads(json.dumps(item["sentiment_analysis"]))
                    if isinstance(item["sentiment_analysis"], list):
                        for sentence in item["sentiment_analysis"]:
                            if sentence and isinstance(sentence, dict):
                                # print(item)
                                if sentence["sentiment_class"] == "positive":
                                    pos_sent += 1
                                    str1 = "<{}>{}</{}>".format(
                                        sentence["sentiment_class"],
                                        sentence["text"],
                                        sentence["sentiment_class"],
                                    )
                                    item["updated_tweet"] = item["updated_tweet"].replace(
                                        sentence["text"], str1
                                    )
                                elif sentence["sentiment_class"] == "negative":
                                    neg_sent += 1
                                    str1 = "<{}>{}</{}>".format(
                                        sentence["sentiment_class"],
                                        sentence["text"],
                                        sentence["sentiment_class"],
                                    )
                                    item["updated_tweet"] = item["updated_tweet"].replace(
                                        sentence["text"], str1
                                    )
                                else:
                                    str1 = "<{}>{}</{}>".format(
                                        "normal", sentence["text"], "normal"
                                    )
                                    item["updated_tweet"] = item["updated_tweet"].replace(
                                        sentence["text"], str1
                                    )
                except Exception as e:
                    print("Exception++++++",e)
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
            
            
                Events_Detected = []
                Labels = []
                try:
                    if "Top_Events" in item.keys():
                        if item["Top_Events"] != "Null":
                            for i in item["Top_Events"]:
                                Events_Detected.append(i["text"])
                                Labels.append(str(i["labels"]))
                    else:
                        logging.error("Top_Event Key not found in tweets")
                # for i in item["All_Events"]:
                #     percentile = 0.8
                #     threshold = percentile * float(i['probability'][0])
                #     probability_list = [float(probability) for probability in i['probability'] if float(
                #         probability) >= threshold]
                #     probability_list = probability_list[0:3]
                #     total_top_labels = len(probability_list)
                #     top_labels_dict = {
                #         'text': i['text'], 'labels': i['labels'][:total_top_labels], 'probability': probability_list}
                #     Events_Detected.append(top_labels_dict['text'])
                #     Labels.append(str(top_labels_dict['labels']))
                    dict1 = {
                        # "Tweet_Title": item["title"],
                        "Tweet_Sentiment": sentiment_class,
                        "Tweet_Time_Stamp": item["tweet_date"].split()[1],
                        "Tweet_Sentiment_Score": round(sentiment_score, 2),
                        # "Tweet_Summary": item["summary"],
                        "Tweet_Tags": [],
                        "Tweet_Source": item["tweet_by"],
                        "Tweet_ID": index,
                        "ID": item["tweets_ID"],
                        "Events_Detected": Events_Detected,
                        "Labels": Labels,
                        "Actual_Tweet": item["updated_tweet"],
                        "Tweet_Text": item["tweet_text"],
                        "Tweet_date": item["tweet_date"],
                        "Sentiment_Analyzed_Tweet": item["tweet_text"],
                    }
                    data_collect.append(dict1)
                except Exception as e:
                    print("Going in exception for Tweets!!! ->", e)
        logging.info('create_tweets')
        return {"Word_Cloud": word_cloud_freq, "Tweets_List": data_collect}
    except Exception as e:
        print("Exception in create_tweets ", e)


def create_news(ticker_data, all_news):
    try:

        body = ""
        events_json = {}
        # Iterates through all the objects, doing the pagination for you. Each obj
        # is an ObjectSummary, so it doesn't contain the body. You'll need to call
        # get to get the whole body.
        for obj in s3_topic_bucket.objects.all():
                key = obj.key
                if key == "Json/News/{}".format(ticker_data + ".json"):
                    events_json = json.loads(obj.get()["Body"].read().decode('utf-8'))
        word_cloud_freq = {}
        for keys, values in events_json.items():
            word_cloud_freq[keys] = []
            if isinstance(values, str):
                word_cloud_freq[keys] = [
                    {"id": 1, "text": "DATA NOT AVAILABLE", "ids":[]}
                ]
            else:
                cnt = 1
                for key, item in values["Topic_News_ID"].items():
                    word_cloud_freq[keys].append({"id": cnt, "text": key,  "ids":item})

        data_collect = []
        for index, item in enumerate(all_news[ticker_data]):
            str1 = ""
            if (
                item["news_ner_flag"]
                and "sentiment_analysis" in item.keys()
                and "summary" in item.keys()
            ):
                pos_sent, neg_sent = 0, 0
                sentiment_class = ""
                sentiment_score = 0.0
                try:
                    item["sentiment_analysis"] = json.loads(json.dumps(item["sentiment_analysis"]))
                    if isinstance(item["sentiment_analysis"], list):
                        for sentence in item["sentiment_analysis"]:
                            if sentence and isinstance(sentence, dict):
                                # print(sentence)
                                if sentence["sentiment_class"] == "positive":
                                    str1 = "<{}>{}</{}>".format(
                                        sentence["sentiment_class"],
                                        sentence["text"],
                                        sentence["sentiment_class"],
                                    )
                                    item["long_description"] = str(item["long_description"]).replace(
                                        sentence["text"], str1
                                    )
                                    pos_sent += 1
                                elif sentence["sentiment_class"] == "negative":
                                    str1 = "<{}>{}</{}>".format(
                                        sentence["sentiment_class"],
                                        sentence["text"],
                                        sentence["sentiment_class"],
                                    )
                                    item["long_description"] = str(item["long_description"]).replace(
                                        sentence["text"], str1
                                    )
                                    neg_sent += 1
                                else:
                                    str1 = "<{}>{}</{}>".format(
                                        "normal", sentence["text"], "normal"
                                    )
                                    item["long_description"] = str(item["long_description"]).replace(
                                        sentence["text"], str1
                                    )
                except Exception as e:
                    print("Exception create news",e)
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
                Events_Detected = []
                Labels = []
                
                try:
                    if "Top_Events" in item.keys():
                        if item["Top_Events"] != "Null":
                            for i in item["Top_Events"]:
                                Events_Detected.append(i['text'])
                                Labels.append(str(i['labels']))
                    else:
                        logging.error("Top_Events key not found in news")
                    dict1 = {
                        "News_Title": item["title"],
                        "News_Source": item["news_source"],
                        "News_Sentiment": sentiment_class,
                        "News_Time_Stamp": item["news_date"].split()[1],
                        "News_Sentiment_Score": round(sentiment_score, 2),
                        "News_Summary": item["summary"],
                        "News_Tags": [],
                        "News_ID": index,
                        "ID":item["news_ID"],
                        "Events_Detected": Events_Detected,
                        "Labels": Labels,
                        "News": item["long_description"],
                        "Actual_News": item["long_description"],
                        "News_date": item["news_date"],
                        "Sentiment_Analyzed_News": item["long_description"],
                    }
                    
                    data_collect.append(dict1)
                except Exception as e:
                    print("Going in exception for News!!! ->", e)
        logging.info('create_news')
        
        return {"Word_Cloud": word_cloud_freq, "News_List": data_collect}
    except Exception as e:
        print("Exception in create_news ", e)


def create_prediction(ticker_data, today_date, summary_data_days):
    # print(credentials)
    try:
        if (nw >= time1 and nw <= time2) and (str(nw.date()) in list_working_days):
        # if(True):
            # print(credentials)
            cursor.execute(
                "show columns FROM {last5}  ".format(
                    last5=trends_prediction_stock_table)
            )
            # print(cursor.fetchall())
            # print(summary_data_days)
            cursor.execute(
                "select date, output, percentile   FROM {prediction} where  ticker_id = '{ticker_id}' ORDER BY prediction_date desc limit {num} ".format(
                    prediction=trends_prediction_stock_table,
                    ticker_id=ticker_data["Stock_Ticker_Symbol"],
                    num=int(ticker_data["Last_5_day_data"])
                )
            )
            trends_data = {"Records": list(map(list, cursor.fetchall()))}
            filename5 = "{}/trends_data.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename5, trends_data)
            trends_data = trends_data["Records"]
            trends_dic = defaultdict(dict)
            for trends_date, trends_output, trends_prediction in trends_data:
                trends_date = list(map(str, trends_date.split(",")))
                trends_output = list(map(str, trends_output.split(",")))
                trends_prediction = list(map(float, trends_prediction.split(",")))
                for d, o, p in zip(trends_date, trends_output, trends_prediction):
                    p = round(float(p), 2)
                    if o == "nan":
                        # if xbom.is_session(pd.Timestamp(d)):
                        if d in list_working_days:
                            trends_dic[d]["days_5_trend_actual"] = int(p)
                            trends_dic[d]["days_5_trend_predicted"] = "NA"

                    else:
                        # if xbom.is_session(pd.Timestamp(d)):
                        if d in list_working_days:
                            trends_dic[d]["days_5_trend_actual"] = int(p)
                            trends_dic[d]["days_5_trend_predicted"] = "NA"

            # for item in trends_dic.values():
            #     if 'days_5_trend_predicted' not in item.keys():
            #         # print(item)
            cursor.execute(
                "select percentile  FROM {prediction} where  ticker_id = '{ticker_id}' order by prediction_date desc limit 1 ".format(
                    prediction=trends_prediction_stock_table,
                    ticker_id=ticker_data["Stock_Ticker_Symbol"],
                )
            )
            trends_prediction_stock_table_single_data = {
                "Records": list(map(str, cursor.fetchall()[0]))}
            filename6 = "{}/trends_prediction_stock_table_single_data.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename6, trends_prediction_stock_table_single_data)
            trends_prediction_stock_table_single_data = trends_prediction_stock_table_single_data[
                "Records"]
            # print(trends_prediction_stock_table_single_data)

            cursor.execute(
                "select *  FROM {prediction} where  ticker_id = '{ticker_id}' ORDER BY date DESC LIMIT 1".format(
                    prediction=rds_stock_next5Days_prediction_table,
                    ticker_id=ticker_data["Stock_Ticker_Symbol"],
                )
            )
            next5_days_data_cursor = {"Records": list(cursor.fetchall()[0])}
            filename7 = "{}/next5_days_data_cursor.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename7, next5_days_data_cursor)
            next5_days_data_cursor = next5_days_data_cursor["Records"]
            next5_days_data = list(map(str, next5_days_data_cursor))
            next5_days_data_cursor = next5_days_data_cursor[2:]
            # print(next5_days_data_cursor)
            next5_days_data_cursor.insert(1, None)
            # print(next5_days_data_cursor)
            next5_days_data_cursor[2], next5_days_data_cursor[3] = (
                next5_days_data_cursor[3],
                next5_days_data_cursor[2],
            )
            next5_days_result = []
            trend = 0
            for item in trends_prediction_stock_table_single_data:
                val = round(float(item.split(",")[-1]), 2)
                if val > 0.5:
                    trend = int(val)
                else:
                    trend = int(val)

            for i in next5_days_data:
                next5_days_result.append(list(map(str, i.split(",")))[-1])
            next5_days_data = ""
            if float(next5_days_result[3]) >= 0.0:
                next5_days_data = "RISE ({})".format(
                    round(float(next5_days_result[-1]), 2))
            else:
                next5_days_data = "DROP ({})".format(
                    round(float(next5_days_result[-1]), 2))

            cursor.execute(
                "select  last_5_dates, output_last_5_days, confidence_score, prediction_last_5_days FROM  {last5} where ticker_id='{ticker_id}' ORDER BY date DESC limit {num} ".format(
                    last5=rds_stock_last5Days_prediction_table,
                    ticker_id=ticker_data["Stock_Ticker_Symbol"],
                    num=int(ticker_data["Last_5_day_data"])
                )
            )
            last5_days_prediction = [next5_days_data_cursor]
            # print(last5_days_prediction)
            filename8_1 = {"Records": list(map(list, cursor.fetchall()))}
            filename8 = "{}/filename8_1.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename8, filename8_1)
            filename8_1 = filename8_1["Records"]
            last5_days_prediction.extend(filename8_1)
            # print(last5_days_prediction)
            last5_days_prediction_result = defaultdict(dict_output)
            for items in last5_days_prediction:
                # if None in items:
                #     continue
                # print(items)
                dates = list(map(str, items[0].split(",")))
                output = list(map(float, items[1].split(
                    ","))) if items[1] else [None] * 5
                confidence_score = (
                    list(map(float, items[2].split(",")))
                    if items[2]
                    else [0.72, 0.7, 0.54, 0.48, 0.38]
                )
                prediction_output = list(map(float, items[3].split(",")))
                for date, out, conf, pred in zip(
                    dates, output, confidence_score, prediction_output
                ):
                    # if date == '2022-01-24':
                    #     # print(date, out, conf, pred)
                    if out != None:
                        output1 = "RISE" if out >= 0.0 else "DROP"
                    else:
                        output1 = None
                    # if xbom.is_session(pd.Timestamp(date)):
                    if date in list_working_days:
                        last5_days_prediction_result[date] = {
                            "output": output1,
                            "confidence_score": conf,
                            "prediction": "RISE" if pred >= 0.0 else "DROP",
                        }

            cursor.execute(
                "select DATE_FORMAT({prediction}.date, '%Y-%m-%d'), {prediction}.output, {prediction}.confidence_score, {prediction}.actual_output FROM {prediction} where {prediction}.ticker_id='{ticker_id}' ORDER BY {prediction}.date DESC limit {num}".format(
                    prediction=rds_stock_prediction_table,
                    ticker_id=ticker_data["Stock_Ticker_Symbol"],
                    num=int(ticker_data["Predition_Number_Of_Days"])
                )
            )

            # cursor.execute("select DATE_FORMAT({prediction}.date, '%Y-%m-%d'), {prediction}.output, {prediction}.confidence_score, {prediction}.actual_output, {next5}.prediction_next_5_days,  {last5}.prediction_last_5_days, {last5}.output_last_5_days FROM {prediction}, {next5}, {last5} where {prediction}.ticker_id = {next5}.ticker_id and {prediction}.ticker_id = {last5}.ticker_id and  {prediction}.ticker_id='{ticker_id}' and date({prediction}.date) between '{sum_days}' and '{today}'  ".format(prediction= rds_stock_prediction_table, next5=rds_stock_next5Days_prediction_table,last5=rds_stock_last5Days_prediction_table, ticker_id = ticker_data["Stock_Ticker_Symbol"],sum_days=summary_data_days, today = today_date))
            prediction = {"Records": list(map(list, cursor.fetchall()))}
            filename9 = "{}/prediction_RDS.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename9, prediction)
            prediction = prediction["Records"]
            histroy_prediction = []
            # print(trends_dic)
            for i, item in enumerate(prediction):
                val = True
                last5_data = last5_days_prediction_result[item[0]]
                if item[1] != item[3]:
                    val = False
                # if xbom.is_session(pd.Timestamp(item[0])) and item[0] != prediction[0][0]:
                if (item[0] in list_working_days) and (item[0] != prediction[0][0]):
                    dic = {
                        "id": i + 1,
                        "day": item[0],
                        "day_1_predicted": "{} ({})".format(item[1], round(item[2], 2)),
                        "days_1_predicted_status": val,
                        "day_1_actual": item[3],
                        # "days_5_predicted": "{}({})".format(last5_data["prediction"],last5_data["confidence_score"]),
                        # "days_5_predicted_status":last5_data["prediction"] == last5_data["output"],
                        # "day_5_actual":last5_data["output"],
                    }
                    if trends_dic[item[0]]:
                        dic["days_5_trend_actual"] = int(
                            trends_dic[item[0]]["days_5_trend_actual"])
                        dic["days_5_trend_predicted"] = trends_dic[item[0]
                                                                ]["days_5_trend_predicted"]
                    if last5_data["prediction"]:
                        dic["days_5_predicted"] = "{} ({})".format(
                            last5_data["prediction"], round(
                                last5_data["confidence_score"], 2)
                        )
                        if last5_data["output"] != None:
                            dic["days_5_predicted_status"] = (
                                last5_data["prediction"] == last5_data["output"]
                            )
                        else:
                            dic["days_5_predicted_status"] = None
                        dic["day_5_actual"] = last5_data["output"]
                    histroy_prediction.append(dic)
            # print(len(histroy_prediction))
            logging.info('create prediction')
            return {
                "Prediction": {
                    "id": "1",
                    "prediction_for": prediction[0][0],
                    "day_1y_ahead_strategy": prediction[0][1],
                    "day_1_confidence_score": round(prediction[0][2], 2),
                    "day_1_actual_output": prediction[0][3],
                    "days_5_ahead_strategy": next5_days_data,
                    "days_5_trend_detection": int(float(trend)),
                },
                "historical_prediction_accuracy": histroy_prediction,
            }
        
        else:
            # print(credentials)
            filename5 = "{}/trends_data.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            trends_data = read_data_from_s3(default_bucket,filename5)
            trends_data = trends_data["Records"]
            trends_dic = defaultdict(dict)
            for trends_date, trends_output, trends_prediction in trends_data:
                trends_date = list(map(str, trends_date.split(",")))
                trends_output = list(map(str, trends_output.split(",")))
                trends_prediction = list(map(float, trends_prediction.split(",")))
                for d, o, p in zip(trends_date, trends_output, trends_prediction):
                    p = round(float(p), 2)
                    if o == "nan":
                        # if xbom.is_session(pd.Timestamp(d)):
                        if d in list_working_days:
                            trends_dic[d]["days_5_trend_actual"] = int(p)
                            trends_dic[d]["days_5_trend_predicted"] = "NA"

                    else:
                        # if xbom.is_session(pd.Timestamp(d)):
                        if d in list_working_days:
                            trends_dic[d]["days_5_trend_actual"] = int(p)
                            trends_dic[d]["days_5_trend_predicted"] = "NA"

            # for item in trends_dic.values():
            #     if 'days_5_trend_predicted' not in item.keys():
            #         # print(item)
            filename6 = "{}/trends_prediction_stock_table_single_data.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            trends_prediction_stock_table_single_data = read_data_from_s3(default_bucket,
                filename6)
            trends_prediction_stock_table_single_data = trends_prediction_stock_table_single_data[
                "Records"]
            # print(trends_prediction_stock_table_single_data)
            ##trends_prediction_stock_table_single_data = list(map(str, trends_prediction_stock_table_single_data))
            filename7 = "{}/next5_days_data_cursor.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            next5_days_data_cursor = read_data_from_s3(default_bucket,filename7)
            next5_days_data_cursor = next5_days_data_cursor["Records"]
            # print(next5_days_data_cursor)
            # print(type(next5_days_data_cursor))
            next5_days_data = list(map(str, next5_days_data_cursor))
            next5_days_data_cursor = next5_days_data_cursor[2:]
            # print(next5_days_data_cursor)
            next5_days_data_cursor.insert(1, None)
            # print(next5_days_data_cursor)
            next5_days_data_cursor[2], next5_days_data_cursor[3] = (
                next5_days_data_cursor[3],
                next5_days_data_cursor[2],
            )
            next5_days_result = []
            trend = 0
            for item in trends_prediction_stock_table_single_data:
                val = round(float(item.split(",")[-1]), 2)
                if val > 0.5:
                    trend = int(val)
                else:
                    trend = int(val)

            for i in next5_days_data:
                next5_days_result.append(list(map(str, i.split(",")))[-1])
            next5_days_data = ""
            if float(next5_days_result[3]) >= 0.0:
                next5_days_data = "RISE ({})".format(
                    round(float(next5_days_result[-1]), 2))
            else:
                next5_days_data = "DROP ({})".format(
                    round(float(next5_days_result[-1]), 2))

            last5_days_prediction = [next5_days_data_cursor]
            # print(last5_days_prediction)
            filename8 = "{}/filename8_1.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            filename8_1 = read_data_from_s3(default_bucket,filename8)
            filename8_1 = filename8_1["Records"]
            last5_days_prediction.extend(filename8_1)
            # print(last5_days_prediction)
            last5_days_prediction_result = defaultdict(dict_output)
            for items in last5_days_prediction:
                # if None in items:
                #     continue
                # print(items)
                dates = list(map(str, items[0].split(",")))
                output = list(map(float, items[1].split(
                    ","))) if items[1] else [None] * 5
                confidence_score = (
                    list(map(float, items[2].split(",")))
                    if items[2]
                    else [0.72, 0.7, 0.54, 0.48, 0.38]
                )
                prediction_output = list(map(float, items[3].split(",")))
                for date, out, conf, pred in zip(
                    dates, output, confidence_score, prediction_output
                ):
                    # if date == '2022-01-24':
                    #     # print(date, out, conf, pred)
                    if out != None:
                        output1 = "RISE" if out >= 0.0 else "DROP"
                    else:
                        output1 = None
                    # if xbom.is_session(pd.Timestamp(date)):
                    if date in list_working_days:
                        last5_days_prediction_result[date] = {
                            "output": output1,
                            "confidence_score": conf,
                            "prediction": "RISE" if pred >= 0.0 else "DROP",
                        }

            # cursor.execute("select DATE_FORMAT({prediction}.date, '%Y-%m-%d'), {prediction}.output, {prediction}.confidence_score, {prediction}.actual_output, {next5}.prediction_next_5_days,  {last5}.prediction_last_5_days, {last5}.output_last_5_days FROM {prediction}, {next5}, {last5} where {prediction}.ticker_id = {next5}.ticker_id and {prediction}.ticker_id = {last5}.ticker_id and  {prediction}.ticker_id='{ticker_id}' and date({prediction}.date) between '{sum_days}' and '{today}'  ".format(prediction= rds_stock_prediction_table, next5=rds_stock_next5Days_prediction_table,last5=rds_stock_last5Days_prediction_table, ticker_id = ticker_data["Stock_Ticker_Symbol"],sum_days=summary_data_days, today = today_date))
            filename9 = "{}/prediction_RDS.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            prediction = read_data_from_s3(default_bucket,filename9)
            prediction = prediction["Records"]
            histroy_prediction = []
            # print(trends_dic)
            for i, item in enumerate(prediction):
                val = True
                last5_data = last5_days_prediction_result[item[0]]
                if item[1] != item[3]:
                    val = False
                # if xbom.is_session(pd.Timestamp(item[0])) and item[0] != prediction[0][0]:
                if (item[0] in list_working_days) and (item[0] != prediction[0][0]):
                    dic = {
                        "id": i + 1,
                        "day": item[0],
                        "day_1_predicted": "{} ({})".format(item[1], round(item[2], 2)),
                        "days_1_predicted_status": val,
                        "day_1_actual": item[3],
                        # "days_5_predicted": "{}({})".format(last5_data["prediction"],last5_data["confidence_score"]),
                        # "days_5_predicted_status":last5_data["prediction"] == last5_data["output"],
                        # "day_5_actual":last5_data["output"],
                    }
                    if trends_dic[item[0]]:
                        dic["days_5_trend_actual"] = int(
                            trends_dic[item[0]]["days_5_trend_actual"])
                        dic["days_5_trend_predicted"] = trends_dic[item[0]
                                                                ]["days_5_trend_predicted"]
                    if last5_data["prediction"]:
                        dic["days_5_predicted"] = "{} ({})".format(
                            last5_data["prediction"], round(
                                last5_data["confidence_score"], 2)
                        )
                        if last5_data["output"] != None:
                            dic["days_5_predicted_status"] = (
                                last5_data["prediction"] == last5_data["output"]
                            )
                        else:
                            dic["days_5_predicted_status"] = None
                        dic["day_5_actual"] = last5_data["output"]
                    histroy_prediction.append(dic)
            # print(len(histroy_prediction))
            return {
                "Prediction": {
                    "id": "1",
                    "prediction_for": prediction[0][0],
                    "day_1y_ahead_strategy": prediction[0][1],
                    "day_1_confidence_score": round(prediction[0][2], 2),
                    "day_1_actual_output": prediction[0][3],
                    "days_5_ahead_strategy": next5_days_data,
                    "days_5_trend_detection": int(float(trend)),
                },
                "historical_prediction_accuracy": histroy_prediction,
            }
    except Exception as e:
        print("Exception in create_prediction ", e)


def create_technical_analysis(
    ticker_data, summary_data_days, today_date, all_news, all_tweets, sect_dict, sector
):
    # Fetching the data from rds table
    try:

        # News Sentiment vs Volume Start and total news vs ticker news
        total_news = []
        total_vs_ticker_news = defaultdict(updated_dict)

        for keys, items in all_news.items():
            for item in items:
                date = item["news_date"].split()[0]
                if keys in sect_dict[sector]:
                    total_vs_ticker_news[date]["total_news"] += 1.0
                if keys == ticker_data["Ticker_Name"]:
                    total_vs_ticker_news[date]["news_ticker"] += 1.0

        res_total_vs_ticker_news = defaultdict(list)
        total_vs_ticker_news = OrderedDict(total_vs_ticker_news)
        for key, item in sorted(total_vs_ticker_news.items()):
            res_total_vs_ticker_news["total_news"].append(
                {"date": key.replace("-", ","),
                "total_news": int(item["total_news"])}
            )
            res_total_vs_ticker_news["news_ticker"].append(
                {"date": key.replace("-", ","),
                "news_ticker": int(item["news_ticker"])}
            )
            res_total_vs_ticker_news["date"].append(key.replace("-", ","))

        sentiment_vs_volume = defaultdict(updated_dict)
        sent_news_max = 0
        sent_news_min = 0
        for item in all_news[ticker_data["Ticker_Name"]]:
            date = item["news_date"].split()[0]
            sentiment_vs_volume[date]["news"] += 1
            if item["news_ner_flag"] and "sentiment_analysis" in item.keys():
                # item["sentiment_analysis"] = item["sentiment_analysis"]
                try:
                    for sentence in item["sentiment_analysis"]:
                        #print("sentence->",sentence[0])
                        if sentence:
                            sentiment_vs_volume[date]["cnt_pos_sent"] += 1
                            sentiment_vs_volume[date]["pos_sent"] += round(
                                float(sentence["positive_sentiment_score"]), 2
                            )
                            sent_news_max = max(
                                sent_news_max, float(
                                    sentence["positive_sentiment_score"])
                            )
                            sentiment_vs_volume[date]["cnt_neg_sent"] += 1
                            sentiment_vs_volume[date]["neg_sent"] += round(
                                float(sentence["negative_sentiment_score"]), 2
                            )
                            sent_news_max = max(
                                sent_news_max, float(
                                    sentence["negative_sentiment_score"])
                            )
                            sentiment_vs_volume[date]["cnt_neu_sent"] += 1
                            sentiment_vs_volume[date]["neu_sent"] += round(
                                float(sentence["neutral_sentiment_score"]), 2
                            )
                            sent_news_max = max(
                                sent_news_max, float(
                                    sentence["neutral_sentiment_score"])
                            )
                except Exception as e:
                    print("Exception ++++ tech",e)
        sentiment_vs_volume = OrderedDict(sentiment_vs_volume)
        res_sentiment_vs_volume = defaultdict(list)
        res_sentiment_vs_volume["sent_news_max"] = sent_news_max
        res_sentiment_vs_volume["sent_news_min"] = sent_news_min
        for key, item in sorted(sentiment_vs_volume.items()):
            if item["cnt_pos_sent"]:
                res_sentiment_vs_volume["pos_sent"].append(
                    {
                        "date": key,
                        "value": round(item["pos_sent"] / item["cnt_pos_sent"], 2)
                    }
                )
            else:
                res_sentiment_vs_volume["pos_sent"].append({
                    "date": key,
                    "value": 0.0
                })
            if item["cnt_neg_sent"]:
                res_sentiment_vs_volume["neg_sent"].append(
                    {
                        "date": key,
                        "value": round(item["neg_sent"] / item["cnt_neg_sent"], 2)
                    }
                )
            else:
                res_sentiment_vs_volume["neg_sent"].append({
                    "date": key,
                    "value": 0.0
                })
            if item["cnt_neu_sent"]:
                res_sentiment_vs_volume["neu_sent"].append(
                    {
                        "date": key,
                        "value": round(item["neu_sent"] / item["cnt_neu_sent"], 2)
                    }
                )
            else:
                res_sentiment_vs_volume["neu_sent"].append({
                    "date": key,
                    "value": 0.0
                })
            res_sentiment_vs_volume["dates"].append(key)
            res_sentiment_vs_volume["news"].append({
                "date": key,
                "value": int(item["news"])
            })


        # End of News Sentiment vs Volume Start and total news vs ticker news

        # Tweets Sentiment vs Volume Start and total tweets vs ticker tweets
        total_vs_ticker_tweet = defaultdict(updated_dict)

        for keys, items in all_tweets.items():
            for item in items:
                date = item["tweet_date"].split()[0]
                if keys in sect_dict[sector]:
                    total_vs_ticker_tweet[date]["total_tweet"] += 1.0
                if keys == ticker_data["Ticker_Name"]:
                    total_vs_ticker_tweet[date]["tweet_ticker"] += 1.0

        total_vs_ticker_tweet = OrderedDict(total_vs_ticker_tweet)
        res_total_vs_ticker_tweet = defaultdict(list)
        for key, item in sorted(total_vs_ticker_tweet.items()):
            res_total_vs_ticker_tweet["total_tweet"].append(
                {"date": key.replace("-", ","),
                "total_tweet": int(item["total_tweet"])}
            )
            res_total_vs_ticker_tweet["tweet_ticker"].append(
                {"date": key.replace("-", ","),
                "tweet_ticker": int(item["tweet_ticker"])}
            )
            res_total_vs_ticker_tweet["date"].append(key.replace("-", ","))

        tweet_sentiment_vs_volume = defaultdict(updated_dict)
        sent_tweet_max = 0
        sent_tweet_min = 0
        for item in all_tweets[ticker_data["Ticker_Name"]]:
            date = item["tweet_date"].split()[0]
            tweet_sentiment_vs_volume[date]["tweet"] += 1
            if item["tweet_ner_flag"] and "sentiment_analysis" in item.keys():
                try:
                    for sentence in item["sentiment_analysis"]:
                        
                        if sentence:
                            tweet_sentiment_vs_volume[date]["cnt_pos_sent"] += 1
                            tweet_sentiment_vs_volume[date]["pos_sent"] += round(
                            float(sentence["positive_sentiment_score"]), 2
                            )
                            sent_tweet_max = max(
                                sent_tweet_max, float(
                                    sentence["positive_sentiment_score"])
                            )
                            tweet_sentiment_vs_volume[date]["cnt_neg_sent"] += 1
                            tweet_sentiment_vs_volume[date]["neg_sent"] += round(
                                float(sentence["negative_sentiment_score"]), 2
                            )
                            sent_tweet_max = max(
                                sent_tweet_max, float(
                                    sentence["negative_sentiment_score"])
                            )
                            tweet_sentiment_vs_volume[date]["cnt_neu_sent"] += 1
                            tweet_sentiment_vs_volume[date]["neu_sent"] += round(
                                float(sentence["neutral_sentiment_score"]), 2
                            )
                            sent_tweet_max = max(
                                sent_tweet_max, float(
                                    sentence["neutral_sentiment_score"])
                            )
                except Exception as e:
                    print("Exception+++++ new tech",e)
        res_tweet_sentiment_vs_volume = defaultdict(list)
        tweet_sentiment_vs_volume = OrderedDict(tweet_sentiment_vs_volume)
        for key, item in sorted(tweet_sentiment_vs_volume.items()):
            if item["cnt_pos_sent"]:
                res_tweet_sentiment_vs_volume["pos_sent"].append(
                    {
                        "date": key,
                        "value": round(item["pos_sent"] / item["cnt_pos_sent"], 2)
                    }
                )
            else:
                res_tweet_sentiment_vs_volume["pos_sent"].append({
                    "date": key,
                    "value": 0.0
                })
            if item["cnt_neg_sent"]:
                res_tweet_sentiment_vs_volume["neg_sent"].append(
                    {
                        "date": key,
                        "value": round(item["neg_sent"] / item["cnt_neg_sent"], 2)
                    }
                )
            else:
                res_tweet_sentiment_vs_volume["neg_sent"].append({
                    "date": key,
                    "value": 0.0
                })
            if item["cnt_neu_sent"]:
                res_tweet_sentiment_vs_volume["neu_sent"].append(
                    {
                        "date": key,
                        "value": round(item["neu_sent"] / item["cnt_neu_sent"], 2)
                    }
                )
            else:
                res_tweet_sentiment_vs_volume["neu_sent"].append(
                    {
                        "date": key,
                        "value": 0.0
                    }
                )
            res_tweet_sentiment_vs_volume["dates"].append(key)
            res_tweet_sentiment_vs_volume["tweets"].append({
                "date": key,
                "value": int(item["tweet"])
            })
        res_tweet_sentiment_vs_volume["sent_tweet_max"] = sent_tweet_max
        res_tweet_sentiment_vs_volume["sent_tweet_min"] = sent_tweet_min

        # End of News Sentiment vs Volume Start and total news vs ticker news

        # Stock Candle Stick and close vs volume
        candle_stick_data = []
        close_vs_volume = defaultdict(list)
        cnt = 1
        max1_close = 0
        min1_close = 0
        if (nw >= time1 and nw <= time2) and (str(nw.date()) in list_working_days):
        # if(True):
            cursor.execute(
                "select open, close, high, low, volume, DATE_FORMAT(date, '%Y-%m-%d')  from {table} where ticker_id='{symbol}' and DATE_FORMAT(date, '%Y-%m-%d') between '{summary}' and '{today}' and DATE_FORMAT(date, '%H') >= '{close_time}'".format(
                    table=rds_stock_data_table,
                    symbol=ticker_data["Stock_Ticker_Symbol"],
                    summary=summary_data_days,
                    today=today_date,
                    close_time="15",
                )
            )
            candle_stick = {"Records": list(map(list, cursor.fetchall()))}
            # removing duplicate records from the fetched tickers data
            candle_final={"Records":[]}
            check_map={}
            for x in candle_stick['Records']:
                if x[5] not in check_map:
                    check_map[x[5]] = True
                    candle_final['Records'].append(x)
            candle_stick = candle_final
            filename10 = "{}/candle_stick.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename10, candle_stick)
            candle_stick = candle_stick["Records"]
            for open1, close, high, low, volume, date in candle_stick:
                if date in list_working_days:
                    candle_stick_data.append(
                        {
                            "id": cnt,
                            "x": date,
                            "y": [open1, high, low, close],
                        }
                    )
                    max1_close = max(max1_close, close)
                    close_vs_volume["date"].append(date)
                    close_vs_volume["close"].append({
                        "date": date,
                        "value": close
                    })
                    close_vs_volume["volume"].append({
                        "date": date,
                        "value": int(volume)
                    })
                    cnt += 1
            close_vs_volume["max1_close"] = max1_close
            close_vs_volume["min1_close"] = min1_close
            logging.info('create technical analysis')
            # End of Stock Candle Stick
            return {
                "Stock_Candle_Stick_Data": candle_stick_data,
                "Stock_Candle_Stick_Data_y_format": "[open, high, low, close]",
                "close_vs_volume": close_vs_volume,
                "news_count_vs_total_news": res_total_vs_ticker_news,
                "news_sentiment_vs_volume": res_sentiment_vs_volume,
                "tweet_count_vs_total_tweet": res_total_vs_ticker_tweet,
                "tweet_sentiment_vs_volume": res_tweet_sentiment_vs_volume,
            }

        else:
            filename10 = "{}/candle_stick.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            candle_stick = read_data_from_s3(default_bucket,filename10)
            candle_stick = candle_stick["Records"]
            for open1, close, high, low, volume, date in candle_stick:
                if date in list_working_days:
                    candle_stick_data.append(
                        {
                            "id": cnt,
                            "x": date,
                            "y": [open1, high, low, close],
                        }
                    )
                    max1_close = max(max1_close, close)
                    close_vs_volume["date"].append(date)
                    close_vs_volume["close"].append({
                        "date": date,
                        "value": close
                    })
                    close_vs_volume["volume"].append({
                        "date": date,
                        "value": int(volume)
                    })
                    cnt += 1
            close_vs_volume["max1_close"] = max1_close
            close_vs_volume["min1_close"] = min1_close
            # End of Stock Candle Stick
            return {
                "Stock_Candle_Stick_Data": candle_stick_data,
                "Stock_Candle_Stick_Data_y_format": "[open, high, low, close]",
                "close_vs_volume": close_vs_volume,
                "news_count_vs_total_news": res_total_vs_ticker_news,
                "news_sentiment_vs_volume": res_sentiment_vs_volume,
                "tweet_count_vs_total_tweet": res_total_vs_ticker_tweet,
                "tweet_sentiment_vs_volume": res_tweet_sentiment_vs_volume,
            }
    except Exception as e:
        print("Error occured in crete technical analysis", e)


# Create Header Data
def create_header(ticker_data, list_months, today_date, index):
    try:
    
        if (nw >= time1 and nw <= time2) and (str(nw.date()) in list_working_days):
        # if(True):
            Last_Updated = (dt.datetime.now(dt.timezone.utc) +
                            dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M")
            Last_Updated_Stocks = {"Records": Last_Updated}
            filename11 = "{}/Last_Updated_Stocks.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename11, Last_Updated_Stocks)
            return_data = {
                "Stock_ID": index,
                "Stock_Name": ticker_data["Ticker_Name"],
                "Stock_Symbol": ticker_data["Stock_Ticker_Symbol"],
                "Stock_Sector": ticker_data["Sector"],
                "Last_Updated": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M"),
                "Last_Updated_Stocks": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=28)).strftime("%Y-%m-%d %H:%M"),
            }
            date_days_ago = today_date - dt.timedelta(days=1)
            if calendar.day_name[date_days_ago.weekday()] in ["Saturday", "Sunday"]:
                date_days_ago = (
                    f"{((dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)) + relativedelta(weekday=FR(-1))):%Y-%m-%d}"
                )
            cursor.execute(
                "select Open, Close, Volume, date  from {} where ticker_id='{}' and DATE_FORMAT(date, '%H') >= '{}' ORDER BY date DESC LIMIT 1".format(
                    rds_stock_data_table,
                    ticker_data["Stock_Ticker_Symbol"],
                    "15",
                )
            )
            yesterday = {"Records": list(map(list, cursor.fetchall()))}
            print(type(yesterday['Records'][0][3].strftime('%Y-%m-%d %H:%M:%S')))
            yesterday['Records'][0][3]=yesterday['Records'][0][3].strftime('%Y-%m-%d %H:%M:%S')
            filename12 = "{}/yesterday2.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            
            print("yesterday->",yesterday)
            write_json_file(filename12, yesterday)

            yesterday = yesterday["Records"]
            cursor.execute(
                "select Open, Close, Volume, date  from {} where ticker_id='{}' ORDER BY date DESC LIMIT 1".format(
                    rds_stock_data_table, ticker_data["Stock_Ticker_Symbol"]
                )
            )
            today = {"Records": list(map(list, cursor.fetchall()))}
            filename13 = "{}/today.json".format(ticker_data["Stock_Ticker_Symbol"])
            write_json_file(filename13, today)
            today = today["Records"]

            if len(today) == 0 and len(yesterday) == 0:
                return_data["Yesterdays_Close"] = "Data not Present"
                return_data["Todays_Open"] = "Data not Present"
                return_data["Todays_Close"] = "Data not Present"
            elif len(today) == 0:
                return_data["Todays_Open"] = "Data not Present"
                return_data["Todays_Close"] = "Data not Present"
                return_data["Yesterdays_Close"] = round(float(yesterday[0][1]), 2)
            elif len(yesterday) == 0:
                return_data["Yesterdays_Close"] = "Data not Present"
                return_data["Todays_Open"] = round(float(today[-1][0]), 2)
                return_data["Todays_Close"] = round(float(today[-1][1]), 2)
            else:
                return_data["Yesterdays_Close"] = round(float(yesterday[0][1]), 2)
                return_data["Todays_Open"] = round(float(today[-1][0]), 2)
                return_data["Todays_Close"] = round(float(today[-1][1]), 2)
            logging.info('create header data')
            return return_data

        else:
            filename11 = "{}/Last_Updated_Stocks.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            Last_Updated_Stocks = read_data_from_s3(default_bucket,filename11)
            Last_Updated_Stocks = Last_Updated_Stocks["Records"]
            return_data = {
                "Stock_ID": index,
                "Stock_Name": ticker_data["Ticker_Name"],
                "Stock_Symbol": ticker_data["Stock_Ticker_Symbol"],
                "Stock_Sector": ticker_data["Sector"],
                "Last_Updated": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M"),
                "Last_Updated_Stocks": Last_Updated_Stocks,
            }
            date_days_ago = today_date - dt.timedelta(days=1)
            if calendar.day_name[date_days_ago.weekday()] in ["Saturday", "Sunday"]:
                date_days_ago = (
                    f"{((dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)) + relativedelta(weekday=FR(-1))):%Y-%m-%d}"
                )
            filename12 = "{}/yesterday2.json".format(
                ticker_data["Stock_Ticker_Symbol"])
            yesterday = read_data_from_s3(default_bucket,filename12)
            yesterday = yesterday["Records"]
            filename13 = "{}/today.json".format(ticker_data["Stock_Ticker_Symbol"])
            today = read_data_from_s3(default_bucket,filename13)
            today = today["Records"]

            if len(today) == 0 and len(yesterday) == 0:
                return_data["Yesterdays_Close"] = "Data not Present"
                return_data["Todays_Open"] = "Data not Present"
                return_data["Todays_Close"] = "Data not Present"
            elif len(today) == 0:
                return_data["Todays_Open"] = "Data not Present"
                return_data["Todays_Close"] = "Data not Present"
                return_data["Yesterdays_Close"] = round(float(yesterday[0][1]), 2)
            elif len(yesterday) == 0:
                return_data["Yesterdays_Close"] = "Data not Present"
                return_data["Todays_Open"] = round(float(today[-1][0]), 2)
                return_data["Todays_Close"] = round(float(today[-1][1]), 2)
            else:
                return_data["Yesterdays_Close"] = round(float(yesterday[0][1]), 2)
                return_data["Todays_Open"] = round(float(today[-1][0]), 2)
                return_data["Todays_Close"] = round(float(today[-1][1]), 2)

            return return_data
    except Exception as e:
        print("Error occured in create header", e)



# Create Company Docs
def create_company_docs(ticker_data):

    data_collect = []
    # print("{}/{}/".format("EarningCalls",ticker_data))
    earnings_result = []
    result=s3_client.list_objects(Bucket=pdf_topic_bucket,Prefix="{}/{}/".format("EarningCalls",ticker_data), Delimiter='/')  #s3_client.list_objects(Bucket=pdf_topic_bucket,Prefix="{}/{}/".format("EarningCalls",ticker_data), Delimiter='/')
    QuarterlyReports_result = s3_client.list_objects(Bucket=pdf_topic_bucket,Prefix="{}/{}/".format("QuarterlyReports",ticker_data), Delimiter='/') #s3_client.list_objects(Bucket=pdf_topic_bucket,Prefix="{}/{}/".format("QuarterlyReports",ticker_data), Delimiter='/')
    # print(result)
    if result.get('CommonPrefixes'):
        for object_summary in result.get('CommonPrefixes'):
            earnings = {}
            earnings["fy"] = "FY{}".format(object_summary.get('Prefix').split('/')[-2])
            earnings["transcripts"] = list()
            earnings["insights"] = list()
            tcnt = 1
            icnt = 1
            theobjects = s3_client.list_objects_v2(Bucket=pdf_topic_bucket, Prefix=object_summary.get('Prefix') ) # s3_client.list_objects_v2(Bucket=pdf_topic_bucket, Prefix=object_summary.get('Prefix') )
            if theobjects:
                for obj in theobjects['Contents']:
                    if obj['Key'].endswith(".pdf"):
                        data = {
                        "id":tcnt,
                        "text":obj['Key'].split("/")[-2],
                        # "url":"https:/{}.s3.ap-south-1.amazonaws.com/{}".format(pdf_topic_bucket,obj['Key'])
                        "url":"https://{}.blob.core.windows.net/{}/{}".format(credentials["BlobAccount"], pdf_topic_bucket, obj['Key'])
                    }
                        earnings["transcripts"].append(data)
                        tcnt +=1
                    elif obj['Key'].endswith('.json'):
                        # result_data = json.loads(s3_client.get_object(Bucket=pdf_topic_bucket, Key=obj['Key'])['Body'].read())
                        result_data = json.loads(s3_client.get_object(Bucket=pdf_topic_bucket, Key=obj['Key'])['Body'].read()) #s3_client.get_object(Bucket=pdf_topic_bucket, Key=obj['Key'])['Body'].read())
                        # print(result_data)
                        data ={
                            "id":tcnt,
                            "text":obj['Key'].split("/")[-2],
                            "Top_Positive_Topics":["Higher Margin","New Products"],
                            "Top_Negative_Topics":["Downtrading","Low Profit"],
                            "Transcript_Analysis":[]
                        }
                        data["Top_Positive_Topics"] = [",".join(i) for i in result_data["file"]["Overall_topics"]["Positive"]]
                        data["Top_Negative_Topics"] = [",".join(i) for i in result_data["file"]["Overall_topics"]["Negative"]]
                        for item in  json.loads(result_data["Output"]["Overall_Documents"]):
                            answer = ""
                            if item["Sentiments"] == "Positive":
                                answer = "<positive>{}</positive>".format(item["Answers"])
                            elif item["Sentiments"] == "Negative":
                                answer = "<negative>{}</negative>".format(item["Answers"])
                            else:
                                answer = "<normal>{}</normal>".format(item["Answers"])
                            data["Transcript_Analysis"].append({
                                "Questions":item["Questions"],
                                "Answers": answer,
                                "doc_sentiment": item["Sentiments"].lower()
                            })
                        earnings["insights"].append(data)
                        icnt += 1
            
            earnings_result.append(earnings)
            
            # print(obj['Key'],object_summary.get('Prefix'))
    QuarterlyReports = defaultdict(list)
    quaterly_annual_reports = []
    if QuarterlyReports_result.get('CommonPrefixes'):
        for object_summary in QuarterlyReports_result.get('CommonPrefixes'):
            QuarterlyReports = {}
            QuarterlyReports["fy"] = "FY{}".format(object_summary.get('Prefix').split('/')[-2])
            QuarterlyReports["Reports"] = list()
            cnt = 1
            theobjects = s3_client.list_objects_v2(Bucket=pdf_topic_bucket, Prefix=object_summary.get('Prefix') )
            if theobjects:
                for obj in theobjects['Contents']:
                    data = {
                        "id":cnt,
                        "text":obj['Key'].split("/")[-2] if "AnnualReport" not in obj['Key'] else "Annual",
                        "url":"https://{}.s3.ap-south-1.amazonaws.com/{}".format(pdf_topic_bucket,obj['Key'])
                    }
                    QuarterlyReports["Reports"].append(data)
                    cnt +=1
            quaterly_annual_reports.append(QuarterlyReports)
    # print(QuarterlyReports)
    logging.info('create company docs')
    return {
        "Earnings":earnings_result,
        "QuarterlyReports": quaterly_annual_reports
    }
        # pass
# debried function for tweets, News, prediction, header, etc
def fetch_debrief_data(
    ticker_data,
    list_months,
    today_date,
    index,
    summary_data_days,
    all_tweets,
    all_news,
    Predition_Number_Of_Days,
    sect_dict,
    sector
):
    header_information = create_header(
        ticker_data, list_months, today_date, index)
    # print("\n\nhead_info fetched: \n\n", header_information, "\n\n")
    overview = create_overview(
        ticker_data, list_months, today_date, index, summary_data_days, all_news, all_tweets)
    # print("\n\noverview fetched: \n\n", overview, "\n\n")
    tweets = create_tweets(ticker_data["Ticker_Name"], all_tweets)
    # print("\n\ntweets fetched: \n\n", tweets, "\n\n")
    news = create_news(ticker_data["Ticker_Name"], all_news)
    # print("\n\nnews fetched: \n\n", news, "\n\n")
    prediction = create_prediction(
        ticker_data, today_date, Predition_Number_Of_Days)
    # print("\n\nprediction fetched: ", prediction, "\n\n")
    technical_analysis = create_technical_analysis(
        ticker_data, summary_data_days, today_date, all_news, all_tweets, sect_dict, sector)
    # print("\n\ntechnical_analysis fetched: \n\n", technical_analysis, "\n\n")
    # company_docs = create_company_docs(ticker_data["Stock_Ticker_Symbol"])
    logging.info('fetch debrief data')
    return {
        "header_information": header_information,
        "overview_tab": overview,
        "tweets": tweets,
        "news": news,
        "prediction": prediction,
        "technical_analysis": technical_analysis,
        # "company_docs": company_docs
    }
