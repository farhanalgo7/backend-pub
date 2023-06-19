
from create_json import working_check, read_data_from_s3
import itertools
from get_dynamodb_data import (
    read_data_dynamodb,
    read_news_data,
    read_tweets_data,
)
# import config
from access_file import get_secret

from collections import defaultdict
from create_json import fetch_data_summary, fetch_debrief_data
from helper import write_json_file
from get_dynamodb_data import read_data_dynamodb
from pytz import timezone
import boto3
import time
import datetime as dt
from datetime import datetime, timedelta, date, timezone
import datetime as dt
import simplejson as json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

logging.warning('main.py')

# from get_rds_data import get_current_stock_data, get_all_stock_data
list_working_days = working_check()
logging.warning('main.py working_check')

start_time = time.time()
""" Connection to dynamodb  """


# dynamodb_client = boto3.resource(
#     "dynamodb",
#     endpoint_url="http://dynamodb.ap-south-1.amazonaws.com",
# )
# s3_client = boto3.client(
#     "s3",
# )
dynamodb_client = boto3.resource(
    "dynamodb",region_name="ap-south-1")
    # endpoint_url="http://dynamodb.ap-south-1.amazonaws.com", aws_access_key_id=config.algo_access_key,aws_secret_access_key=config.algo_secret_access_token,region_name=config.region_name)


s3_client = boto3.client(
    "s3",region_name="ap-south-1")
    # aws_access_key_id=config.algo_access_key,aws_secret_access_key=config.algo_secret_access_token,region_name=config.region_name

logging.warning('main.py Connection to dynamodb and rds instance')

get=get_secret()
credentials = json.loads(get["SecretString"])
default_bucket = credentials["default_bucket"]
# default_bucket = "algoanalytics-fabric-website"

""" Get all the data from config table in dynamodb """

data_config = read_data_dynamodb(dynamodb_client, "ConfigTable")
# print(data_config)
logging.warning('main.py Get all the data from config table in dynamodb')

""" Setting start date and end date for the debrief"""
ct = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)
today_date = ct
# today_date = dt.datetime(2022,9,30,0,0,0).date()
print("utc",dt.datetime.now(dt.timezone.utc) )
print("\n\nct = ",ct, "\n\n" )
# today_date=ct.strftime("%Y-%m-%d %H:%M:%S")

# today_date = dt.datetime(2022,11,22,0,0,0).date()
print("today date ",today_date)
print("this is dynamodb table data")
# print(data_config)
print("*****************")
Debrief_Number_Of_Days = int(data_config[0]["Debrief_Number_Of_Days"])
Summary_Number_Of_Days = int(data_config[0]["Summary_Number_Of_Days"])
Predition_Number_Of_Days = int(data_config[0]["Predition_Number_Of_Days"])
date_days_ago = today_date - dt.timedelta(days=Debrief_Number_Of_Days)

summary_data_days = today_date - dt.timedelta(days=Summary_Number_Of_Days)
prediction_data_days = today_date - dt.timedelta(days=Predition_Number_Of_Days)
print("\nSUMMARY DATA DAYS: ", summary_data_days, "\n\n")

logging.warning('main.py Setting start date and end date for the debrief')

""" Get the data for the overview json file """
overview_file_name = "overview"
overview_json = {
    "Last Updated": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime(
        "%Y-%m-%d %H:%M"
    ),
    "Last Updated Data": [],
}

logging.warning('main.py Get the data for the overview json file')

nw = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)
hrs = nw.hour
mins = nw.minute
secs = nw.second
zero = dt.timedelta(seconds=secs+mins*60+hrs*3600)
st = nw - zero
time1 = st + dt.timedelta(seconds=9*3600+10*60)   # this gives 09:10 AM
time2 = st + dt.timedelta(seconds=17*3600+15*60)  # this gives 05:15 PM
#nw = dt.datetime(2022, 3, 31, 9, 20, 48, 140961, tzinfo=dt.timezone.utc)
print("\n\nmain.py:\n\nNW: ", nw, "\nTIME1: ", time1, "\nTIME2: ", time2, "\n\n")

summary_file_name = "summary.json"

if (nw >= time1 and nw <= time2) and (str(nw.date()) in list_working_days):
    summary_json = {
        "summary": [],
        "Last_Updated": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "Last_Updated_Stocks": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime(
            "%Y-%m-%d %H:%M"
        ),
    }
    Last_Updated_Stocks = {"Records": (dt.datetime.now(
        dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M")}
    filename44 = "Last_Updated_Stocks.json"
    write_json_file(filename44, Last_Updated_Stocks)
    logging.warning('main.py writing summary file')
else:
    filename44 = "Last_Updated_Stocks.json"
    Last_Updated_Stocks = read_data_from_s3(default_bucket,filename44)
    Last_Updated_Stocks = Last_Updated_Stocks["Records"]
    summary_json = {
        "summary": [],
        "Last_Updated": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5, minutes=30)).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "Last_Updated_Stocks": Last_Updated_Stocks
    }

all_news = defaultdict(list)
all_tweets = defaultdict(list)
this_month = (dt.datetime.now(dt.timezone.utc) +
              dt.timedelta(hours=5, minutes=30)).strftime("%Y-%m")
today = dt.date.today()
first = today.replace(day=1)
lastMonth = (first - dt.timedelta(days=1)
             ).strftime("%Y-%m")
list_months = [lastMonth, this_month]
print(list_months)
all_tweets = read_tweets_data(dynamodb_client, summary_data_days, list_months)
all_news = read_news_data(dynamodb_client, summary_data_days, list_months)

# print("All data",all_news , all_tweets)

logging.warning('main.py read tweets and news data')

error_list=[]

def lambd_func(x): return x["Sector"]


logging.warning('main.py lambda function')

data_config_list = [{"Sector": item["Sector"],
                     "Ticker_Name": item["Ticker_Name"]} for item in data_config]
it_stocks_name = itertools.groupby(data_config_list, lambd_func)
# print(list(it_stocks_name))
sect_dict = defaultdict(list)
for key, group in it_stocks_name:
    for item in list(group):
        sect_dict[key].append(item["Ticker_Name"])
print(sect_dict)
logging.warning('main.py sect_dict')
print("size of data_config->",len(data_config_list))
i = 0
for index, data_item in enumerate(data_config):
    try:
        # if data_item["Stock_Ticker_Symbol"] != "ITC.NS":
        #     continue
        # print("index",index)
        # print("data_item ",data_item["Sector"])
        # data_item["Sector"] = "ITC.NS"
        ticker_json = fetch_debrief_data(
            data_item,
            list_months,
            today_date,
            index + 1,
            summary_data_days,
            all_tweets,
            all_news,
            prediction_data_days,
            sect_dict,
            data_item["Sector"]
        )
        
        for item in ticker_json.keys():
            path = "{}/{}.json".format(data_item["Stock_Ticker_Symbol"], item)
            if "header_information" in path:
                ticker_json[item]["Last_Updated"] = summary_json["Last_Updated"]
            print(path)
            write_json_file(path, ticker_json[item])
        results = fetch_data_summary(
            data_item["Ticker_Name"],
            today_date,
            summary_data_days,
            summary_json,
            data_item["Stock_Ticker_Symbol"],
            all_tweets,
            all_news,
        )
        results["id"] = index+1
        results['Sector'] = data_item["Sector"]
        summary_json["summary"].append(results)
        print("file name",summary_file_name)
        print("summary json",summary_json)
       

        logging.warning('main.py fetch_debrief_data')
    except Exception as e:
        print(f'Error occured for {data_item["Stock_Ticker_Symbol"]}')
        error_list.append(data_item["Stock_Ticker_Symbol"])
        print(e)

write_json_file(summary_file_name, summary_json)

print("overview file uploaded")
print(time.time() - start_time)
print(error_list)
