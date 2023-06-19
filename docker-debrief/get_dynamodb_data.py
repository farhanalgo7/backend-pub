from datetime import timedelta
from boto3.dynamodb.conditions import Attr, Key
import time
from collections import defaultdict
import itertools
from pytz import timezone
from datetime import datetime, date, timedelta, timezone

# Read News from dynamoDB 
def read_news_data(dynamodb_client, since, list_months):
    news_table = dynamodb_client.Table("News_Table")
    items = []

    str_dict = {}
    str_dict[":val"] = True
    str_dict[":date"] = since.strftime("%Y-%m-%d 00:00:00")
    for month in list_months:
        time.sleep(1)
        resp_ = news_table.query(
            Limit=75,
            KeyConditionExpression=Key("news_month").eq(month),
            FilterExpression="news_ner_flag= :val AND news_date > :date",
            ExpressionAttributeValues=str_dict,
        )
        items.extend(resp_["Items"])

        while "LastEvaluatedKey" in resp_:
            try:
                time.sleep(1)
                resp_ = news_table.query(
                    Limit=75,
                    ExclusiveStartKey=resp_["LastEvaluatedKey"],
                    KeyConditionExpression=Key("news_month").eq(month),
                    FilterExpression="news_ner_flag= :val AND news_date > :date",
                    ExpressionAttributeValues=str_dict,
                )
                items.extend(resp_["Items"])
            except:
                time.sleep(1)
                resp_ = news_table.query(
                    Limit=75,
                    ExclusiveStartKey=resp_["LastEvaluatedKey"],
                    KeyConditionExpression=Key("news_month").eq(month),
                    FilterExpression="news_ner_flag= :val AND news_date > :date",
                    ExpressionAttributeValues=str_dict,
                )
                items.extend(resp_["Items"])
    news_dict = defaultdict(list)
    for item in items:
        news_dict[item["news_for"]].append(item)
    return news_dict


def read_tweets_data(dynamodb_client, since, list_months):
    news_table = dynamodb_client.Table("Tweets_Table")
    items = []
    tweet_limit = 200
    str_dict = {}
    str_dict[":val"] = True
    str_dict[":date"] = since.strftime("%Y-%m-%d 00:00:00")
    for month in list_months:
        time.sleep(1)
        resp_ = news_table.query(
            Limit=tweet_limit,
            KeyConditionExpression=Key("tweets_month").eq(month),
            FilterExpression="tweet_ner_flag= :val AND tweet_date > :date",
            ExpressionAttributeValues=str_dict,
        )
        items.extend(resp_["Items"])

        while "LastEvaluatedKey" in resp_:
            try:
                time.sleep(1)
                resp_ = news_table.query(
                    Limit=tweet_limit,
                    ExclusiveStartKey=resp_["LastEvaluatedKey"],
                    KeyConditionExpression=Key("tweets_month").eq(month),
                    FilterExpression="tweet_ner_flag= :val AND tweet_date > :date",
                    ExpressionAttributeValues=str_dict,
                )
                items.extend(resp_["Items"])
                tweet_limit = 200
            except:
                time.sleep(1)
                tweet_limit = tweet_limit - 50
                pass
    tweets_dict = defaultdict(list)
    for item in items:
        tweets_dict[item["tweet_for"]].append(item)
    return tweets_dict


def create_default():
    return {"sentiment_score": 0.0, "total": 0}


def read_data_dynamodb(dynamodb_client, table_name):
    """Read whole data from dynamodb table and returns the json object

    Args:
        dynamodb_client : Boto3 DynamoDB client
        table_name (string): Name of the table to read from

    Returns:
        [json]: returns the json object of the table (key-value pairs)
    """

    table = dynamodb_client.Table(table_name)
    response = table.scan()
    data = response["Items"]

    # scan() has 1 MB limit on the amount of data it will return in a request, so we need to paginate through the results in a loop.
    while "LastEvaluatedKey" in response:
        response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        data.extend(response["Items"])

    return data


def sum_get_total_tweets(
    ticker_name, since, until, all_tweets
):
    """
    Get tweets for a given ticker from since to until

    Args:
        ticker_name (string): Ticker name
        since(datetime): Start date
        until(datetime): End date

    Returns:
        dict: Dictionary of tweets for given ticker name in the format of {date: [tweet1, tweet2, ...]}
    """
    required_tweets = {}
    total_tweets = 0
    total_tweet_sentences_sentiment = 0
    total_tweet_sentences_pos_sentiment = 0
    total_tweet_sentences_neg_sentiment = 0
    total_tweet_sentences_neutral_sentiment = 0
    items = []
    for key, item in all_tweets.items():
        if key == ticker_name:
            items = list(item)
            break
        # print(key,len(list(item)))
    total_tweets = len(items)
    delta = until - since
    # for day in range(delta.days + 1):
    #     date = since + timedelta(days=day)
    #     required_tweets[date.strftime("%Y-%m-%d")] = []
    required_tweets = defaultdict(list)

    tweets_sent = defaultdict(create_default)
    sentiment_score = 0.0
    # print("Tweet", len(items))
    for item in items:
        # print(item["tweet_ner_flag"])
        required_tweets[item["tweet_date"].split()[0]].append(item)
        tweets_sent[item["tweet_date"].split()[0]]["total"] += 1
        if item["tweet_ner_flag"] and "sentiment_analysis" in item.keys():
            pos_sent, neg_sent = 0, 0
            total_tweet_sentences_sentiment += len(item["sentiment_analysis"])
            for sentence in item["sentiment_analysis"]:
                if sentence:
                    if sentence["sentiment_class"] == "positive":
                        total_tweet_sentences_pos_sentiment += 1
                        pos_sent += 1
                    elif sentence["sentiment_class"] == "negative":
                        total_tweet_sentences_neg_sentiment += 1
                        neg_sent += 1
                    else:
                        total_tweet_sentences_neutral_sentiment += 1
            if (pos_sent == 0) and (neg_sent == 0):
                sentiment_score = 0.0
            else:

                sentiment_score = (pos_sent - neg_sent) / (pos_sent + neg_sent)
                # print(sentiment_score)
            tweets_sent[item["tweet_date"].split()[0]][
                "sentiment_score"
            ] += sentiment_score

    if tweets_sent[until.strftime("%Y-%m-%d")]["total"]:
        today_tweet_sent = (
            tweets_sent[until.strftime("%Y-%m-%d")]["sentiment_score"]
            / tweets_sent[until.strftime("%Y-%m-%d")]["total"]
        )
    else:
        today_tweet_sent = 0.0
    keys = required_tweets.keys()
    today_tweet = len(required_tweets[until.strftime("%Y-%m-%d")])
    tweet_count = [
        {"date": k, "tweetcount": len(v)} for k, v in required_tweets.items()
    ]
    return (
        tweet_count,
        # required_tweets,
        # total_tweets,
        # total_tweet_sentences_sentiment,
        # total_tweet_sentences_pos_sentiment,
        # total_tweet_sentences_neg_sentiment,
        # total_tweet_sentences_neutral_sentiment,
        tweets_sent,
        today_tweet,
        today_tweet_sent,
    )


# def sum_get_total_news(
#     ticker_name, since, until, all_news
# ):
#     """
#     Get news for a given ticker from since to until

#     Args:
#         ticker_name (string): Ticker name
#         since(datetime): Start date
#         until(datetime): End date

#     Returns:
#         dict: Dictionary of news for given ticker name in the format of {date: [news1, news2, ...]}
#     """
#     required_news = {}
#     # print(until)
#     total_news = 0
#     total_news_sentences_sentiment = 0
#     total_news_sentences_pos_sentiment = 0
#     total_news_sentences_neg_sentiment = 0
#     total_news_sentences_neutral_sentiment = 0

#     items = []
#     for key, item in all_news.items():
#         if key == ticker_name:
#             items = list(item)
#             break
#     total_news = len(items)

#     delta = until - since

#     # for day in range(delta.days + 1):
#     #     date = since + timedelta(days=day)
#     #     required_news[date.strftime("%Y-%m-%d")] = []

#     news_sent = defaultdict(create_default)
#     required_news = defaultdict(list)
#     sentiment_score = 0.0
#     # print("News", total_news)
#     for item in items:
#         required_news[item["news_date"].split()[0]].append(item)
#         news_sent[item["news_date"].split()[0]]["total"] += 1
#         if item["news_ner_flag"] and "sentiment_analysis" in item.keys():
#             total_news_sentences_sentiment += len(item["sentiment_analysis"])
#             pos_sent, neg_sent = 0, 0
#             for sentence in item["sentiment_analysis"]:
#                 if sentence:
#                     # print(sentence)
#                     if sentence["sentiment_class"] == "positive":
#                         total_news_sentences_pos_sentiment += 1
#                         pos_sent += 1
#                     elif sentence["sentiment_class"] == "negative":
#                         total_news_sentences_neg_sentiment += 1
#                         neg_sent += 1
#                     else:
#                         total_news_sentences_neutral_sentiment += 1
#             if (pos_sent == 0) and (neg_sent == 0):
#                 sentiment_score = 0.0
#             else:

#                 sentiment_score = (pos_sent - neg_sent) / (pos_sent + neg_sent)
#             news_sent[item["news_date"].split()[0]][
#                 "sentiment_score"
#             ] += sentiment_score
#     if news_sent[until.strftime("%Y-%m-%d")]["total"]:
#         today_news_sent = (
#             news_sent[until.strftime("%Y-%m-%d")]["sentiment_score"]
#             / news_sent[until.strftime("%Y-%m-%d")]["total"]
#         )
#     else:
#         today_news_sent = 0.0
#     today_news = len(required_news[until.strftime("%Y-%m-%d")])
#     news_count = [{"date": k, "newscount": len(v)} for k, v in required_news.items()]
#     return (
#         news_count,
#         # required_news,
#         # total_news,
#         # total_news_sentences_sentiment,
#         # total_news_sentences_pos_sentiment,
#         # total_news_sentences_neg_sentiment,
#         # total_news_sentences_neutral_sentiment,
#         news_sent,
#         today_news,
#         today_news_sent,
#     )

#for 24hr news volume and news sentiment
def sum_get_total_news(ticker_name, since, until, all_news):
    """
     Get news for a given ticker from since to until

     Args:
         ticker_name (string): Ticker name
         since(datetime): Start date
         until(datetime): End date

     Returns:
         dict: Dictionary of news for given ticker name in the format of {date: [news1, news2, ...]}
     """
    
    total_news_sentences_sentiment = 0
    total_news_sentences_pos_sentiment = 0
    total_news_sentences_neg_sentiment = 0
    total_news_sentences_neutral_sentiment = 0
    
    items = []
    for key, value in all_news.items():
        if key == ticker_name:
            items = list(value)
    
    yesterday = until - timedelta(hours=24)
    
    news_sent = defaultdict(create_default)
    required_today_news_count = defaultdict(list)
    required_news_count = defaultdict(list)
    sentiment_score = 0.0

    for item in items:
        required_today_news_count[item["news_date"]].append(item)
        required_news_count[item["news_date"].split()[0]].append(item)
        news_sent[item["news_date"]]["total"] += 1
            
        if item["news_ner_flag"] and "sentiment_analysis" in item.keys():
            total_news_sentences_sentiment += len(item["sentiment_analysis"])
            pos_sent, neg_sent = 0, 0
                
            try:
                for sentence in item["sentiment_analysis"]:
                    if sentence:
                        # print(sentence)
                        if sentence["sentiment_class"] == "positive":
                            total_news_sentences_pos_sentiment += 1
                            pos_sent += 1
                        elif sentence["sentiment_class"] == "negative":
                            total_news_sentences_neg_sentiment += 1
                            neg_sent += 1
                        else:
                            total_news_sentences_neutral_sentiment += 1
                                
            except Exception as e:
                print("Exception", e)
                    
            if (pos_sent == 0) and (neg_sent == 0):
                sentiment_score = 0.0
            else:
                sentiment_score = (pos_sent - neg_sent) / (pos_sent + neg_sent)
                
            news_sent[item["news_date"]]["sentiment_score"] += sentiment_score
                
    news_count = [{"date": k, "newscount": len(v)} for k, v in required_news_count.items()]
    
    total_sent_score = 0
    total = 0
    for key, value in news_sent.items():
        dt_obj = datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        if dt_obj <= until and dt_obj >= yesterday:
            total_sent_score += value['sentiment_score']
            total += value['total']
    if total:
        today_news_sent = round(total_sent_score / total, 2)
    else:
        today_news_sent = 0.0
        
    today_news_count = 0
    for key in list(required_today_news_count.keys()):
        dt_obj = datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        if dt_obj <= until and dt_obj >= yesterday:
            today_news_count += 1

    return (news_count, news_sent, today_news_count, today_news_sent)
