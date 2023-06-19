import boto3
from boto3.dynamodb.conditions import Key, Attr
from news_ingestion_aws import crawl_news_by_keyword
from tweets_ingestion_aws import crawl_tweets_by_keyword
import time

s = time.perf_counter()

#Initializing DynamoDB Object variable
dynamodb = boto3.resource('dynamodb',endpoint_url='http://dynamodb.ap-south-1.amazonaws.com', region_name="ap-south-1")

#Initializing Dynamodb Table object variables
config_table = dynamodb.Table('ConfigTable')
tweets_table = dynamodb.Table('Tweets_Table')
news_table = dynamodb.Table('News_Table')

#Get all of the items from the Config_Table
config_response = config_table.scan()

for item in config_response["Items"]:
	# print(item)
	# try:
	# 	crawl_tweets_by_keyword(tweets_table, 
	# 			                item['Ticker_Twitter_Keywords'], 
	# 			                item['Exclusion_Tweet_String'],
	# 			                item['Ticker_Name'], 
	# 	                        int(item['Tweet_Followers_Count']), 
	# 			                int(item['Tweet_Minimum_Tokens']), 
	# 			                int(item['Tweet_Minimum_Characters']))	
	# except Exception as e:
	# 	print("Tweet Error occured for ",item['Stock_Ticker_Symbol']," : ",e)

	try:
		crawl_news_by_keyword(news_table, 
							item['News_frequency'], 
							item['Ticker_News_Keywords'], 
							item['Ticker_Name'], 
							item['Sector'])
	except Exception as e:
		print("News Error occured for ",item['Stock_Ticker_Symbol']," : ",e)

elapsed = time.perf_counter() - s
print(f"{__file__} executed in {elapsed:0.2f} seconds.")
