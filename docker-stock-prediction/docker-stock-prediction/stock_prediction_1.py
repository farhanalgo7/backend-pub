"""  Module to do predictions on stocks from various sectors and then insert the prediction table into AWS RDS

1. Get news and stock data from database(dynamoDB)
2. Build feature using - [returns, sector_return, stock_sentiment_score, sector_sentiment_score, volume, sector_volume]
3. Do prediction using Random Forest Classifier
4. Store the output dataframe into AWS RDS'

Prerequisites:
    !pip install boto3
    !pip install pandas
    !pip install numpy
    !pip install sklearn
    !pip install pymysql
    !pip install mysql-connector-python

Contributors:
    Divyank Lunkad <dlunkad@algoanalytics.com>
    Kalash Gandhi <kgandhi@algoanalytics.com>
    Ekansh Gupta <egupta@algoanalytics.com>
    Mrityunjay Samanta <msamanta@algoanalytics.com>

Created: 23rd  December 2021

"""
#----------------------------------- IMPORT SECTION ------------------------------------------------------------
import boto3
from boto3.dynamodb.conditions import Key , Attr
import pandas as pd
import mysql.connector
from datetime import datetime , timezone
import datetime as dt
import time
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.preprocessing import normalize
import ast
from sklearn.model_selection import KFold
import pymysql
import pickle
import sys
import os
import json
from datetime import timezone
os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'
#______________________________________________________________________________________________________________________

#-----------------------------------SECTION - Fetching global variables from Secret manager----------------------------------------------

def get_secret():

    secret_name = "FABRIC-RDS-Secret-Keys"
    region_name = "ap-south-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    return get_secret_value_response

credentials = json.loads(get_secret()['SecretString'])

host = credentials['host']
user= credentials['username']
port= credentials['port']
database= credentials['Database_Name']
region= credentials['region_name']
rds_stock_data_table = credentials['Ingestion_table_name']
rds_stock_prediction_table= credentials['rds_stock_prediction_table']


#____________________________________________________________________________________________________________________________________________

#----------------------------------- SECTION - connection to AWS RDS------------------------------------------------------------

client= boto3.client('rds', region_name=region)
## generating a database authentication token for user mentioned in IAM policy of AWS
try: token = client.generate_db_auth_token(DBHostname=host, Port=port, DBUsername=user, Region=region)
except Exception as err: 
    print("\n could not get authentication token for database, getting error :",err)
    sys.exit()

def get_connection(host , user , token , port , database):
  return mysql.connector.connect(host=host , user=user , password=token , port=port , database=database)

count = 10
## calling get_connection method to get connection object
while(count>0):
    try: 
        db= get_connection( host , user , token , port , database)
        print("Connection Established!! Woohoo!!")
        break
    except: 
        print("\ncould not establish connection with RDS..Retrying.................... Retry Countdown : ",count)
        count-=1
        continue

#______________________________________________________________________________________________________________________


#--------------------------------SECTION - connection to AWS Dynamo DB-------------------------------------------------

def get_ticker_names_and_id(sector , table):
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
    ticker_id=[]
    ticker_name=[]
    response= table.scan(FilterExpression= Attr('Sector').eq(sector))
    for item in response['Items']:
        ticker_id.append(item['Stock_Ticker_Symbol'])
        ticker_name.append(item['Ticker_Name'])
    return ticker_id , ticker_name


def get_news(ticker_list , algoDDB):
    """
    Function to get news of all the companies present in ticker_list

    Parameters:
    ---------
        ticker_list: list
            list of company names
        algoDDB: variable name
            assigned to boto3.resource()
    
    Returns:
    --------
        all_news: dataframe
            dataframe which includes news of all the companies in ticker_list
    """
    this_month = (dt.datetime.now(timezone.utc)+dt.timedelta(hours=5, minutes = 30)).strftime("%Y-%m")
    today = dt.date.today()
    first = today.replace(day=1)
    lastMonth = first - dt.timedelta(days=1)
    last_month=lastMonth.strftime("%Y-%m")
    list_months=[this_month , last_month]
    print('\n list of months to be scanned ',list_months)

    str_dict = {}
    str_dict[":val"]=True
    str_dict[":date"]=(datetime.now(timezone.utc) - dt.timedelta(30)+ dt.timedelta(hours=5 ,minutes=30)).strftime('%Y-%m-%d 00:00:00')
    table=algoDDB.Table('News_Table')
    items=[]
    for month in list_months:
        resp_=table.query(Limit=100,KeyConditionExpression=Key("news_month").eq(month) , 
                        FilterExpression="news_ner_flag= :val AND news_date > :date",
                        ExpressionAttributeValues= str_dict)
        items.extend(resp_['Items'])
    
        while 'LastEvaluatedKey' in resp_:
            try:
                resp_=table.query(Limit=100,ExclusiveStartKey=resp_['LastEvaluatedKey'] ,KeyConditionExpression=Key("news_month").eq(month) , 
                        FilterExpression="news_ner_flag= :val AND news_date > :date",
                        ExpressionAttributeValues= str_dict)
                items.extend(resp_['Items']) 
            except:
                time.sleep(5)
                resp_=table.query(Limit=100,ExclusiveStartKey=resp_['LastEvaluatedKey'] ,KeyConditionExpression=Key("news_month").eq(month) , 
                        FilterExpression="news_ner_flag= :val AND news_date > :date",
                        ExpressionAttributeValues= str_dict)
                items.extend(resp_['Items'])

    all_news = pd.DataFrame(items)
    return all_news

#algoDDB= boto3.resource('dynamodb',region_name=region)
algoDDB= boto3.resource('dynamodb',endpoint_url='http://dynamodb.ap-south-1.amazonaws.com')
table=algoDDB.Table('ConfigTable')

#_____________________________________________________________________________________________________________________


#-----------------------------------GLOBAL variables / dataframes declaration-------------------------------------------

TODAY = dt.datetime.now(timezone.utc)+dt.timedelta(hours=5, minutes = 30)
TODAY_DATE = TODAY.strftime('%Y-%m-%d') # required to check if today is a valid trading day
TOMORROW = TODAY + dt.timedelta(1) 

model_training_duration=20

## get ticker_ids and ticker_names for Bank and IT sector
IT_ticker_id , IT_ticker_name = get_ticker_names_and_id('IT', table)
BANK_ticker_id , BANK_ticker_name = get_ticker_names_and_id('BANK' , table)

bank_placeholder="%s" + ''.join(',%s' * (len(BANK_ticker_id)-1))
it_placeholder="%s" + ''.join(',%s' * (len(IT_ticker_id)-1))

## get live and close price for IT and Bank sector
print("*************** started fetching stock data for IT and BANK sector******************")
#Bankstock_live_price = pd.read_sql("select * from "+rds_stock_data_table+" where ticker_id IN %s AND HOUR(DATE) =14 AND MINUTE(DATE) BETWEEN 30 and 59 order by date",db,params={tuple(BANK_ticker_id)},)
#Bankstock_close_price = pd.read_sql("select * from "+rds_stock_data_table+" where ticker_id IN %s AND HOUR(DATE) > 14 order by date",db,params={tuple(BANK_ticker_id)},)

Bankstock_live_price = pd.read_sql("select * from "+rds_stock_data_table+" where ticker_id IN ("+bank_placeholder+") AND HOUR(DATE) =14 AND MINUTE(DATE) BETWEEN 30 and 59 order by date", db, params=BANK_ticker_id)
Bankstock_close_price = pd.read_sql("select * from "+rds_stock_data_table+" where ticker_id IN ("+bank_placeholder+") AND HOUR(DATE) > 14 order by date",db,params=BANK_ticker_id)

#ITstock_live_price = pd.read_sql("select * from "+rds_stock_data_table+" where ticker_id IN %s AND HOUR(DATE) = 14 AND MINUTE(DATE) BETWEEN 30 and 59 order by date",db, params={tuple(IT_ticker_id)},)
#ITstock_close_price = pd.read_sql( "select * from "+rds_stock_data_table+" where ticker_id IN %s AND HOUR(DATE) > 14 order by date",db,params={tuple(IT_ticker_id)})

ITstock_live_price = pd.read_sql("select * from "+rds_stock_data_table+" where ticker_id IN ("+it_placeholder+") AND HOUR(DATE) = 14 AND MINUTE(DATE) BETWEEN 30 and 59 order by date",db, params=IT_ticker_id)
ITstock_close_price = pd.read_sql( "select * from "+rds_stock_data_table+" where ticker_id IN ("+it_placeholder+") AND HOUR(DATE) > 14 order by date",db,params=IT_ticker_id)

print("*************** completed fetching stock data for IT and BANK sector******************")

it_latest_date = ITstock_close_price.sort_values(by='date' , ascending=False).reset_index()['date'][0].strftime('%Y-%m-%d')
bank_latest_date= Bankstock_close_price.sort_values(by='date' , ascending=False).reset_index()['date'][0].strftime('%Y-%m-%d')
it_output= pd.DataFrame()
bank_output = pd.DataFrame()
## get IT and Bank news
print("*************** started fetching IT and BANK news**********************************")
News = get_news(BANK_ticker_name+IT_ticker_name , algoDDB)
Bank_News= News[News['news_for'].isin(BANK_ticker_name)]
IT_News=   News[News['news_for'].isin(IT_ticker_name)]
# Bank_News = get_news(BANK_ticker_name , algoDDB)
# IT_News=get_news(IT_ticker_name , algoDDB)
print("*************** completed fetching IT  and BANK news**********************************")

#________________________________________________________________________________________________________________________



#----------------------------------------------- SECTION - Functions declaration----------------------------------------

def build_sentiment_mapping_csv(db_resource, sector):
    """
    Function to build sentiment mapping csv 
    
    Parameters
    ---------
        db_resource: dynamodb resource object
            assigned to boto3.resource()
        sector: str
            name of sector - IT or BANK

    """
    ticker_sentiment_mapping = pd.DataFrame(columns=["news_for", "ticker_id", "keywords", "ticker_name"])
    config_table = db_resource.Table("ConfigTable")
    response = config_table.scan(FilterExpression=Attr("Sector").eq(sector))
    for item in response["Items"]:
        ticker_sentiment_mapping = ticker_sentiment_mapping.append(
            {
                "news_for": item["Ticker_Name"],
                "ticker_id": item["Stock_Ticker_Symbol"],
                "ticker_name": item["Ticker_Sentiment_Mapping"]["SS"],
            },
            ignore_index=True,
        )
    return ticker_sentiment_mapping

def build_returns_and_output(stock_prices):
    """
    Calculate returns and output

    Parameters
    ---------
    stock_prices : DataFrame
        numerical data about the stocks
    """
    stock_prices['log_close'] = np.log(stock_prices['close'])
    stock_prices.sort_values(['date','ticker_id'], inplace = True)
    stock_prices['returns'] = stock_prices.groupby('ticker_id')['log_close'].diff()
    stock_prices['output'] = -(stock_prices.groupby('ticker_id')['log_close'].diff(-1))
    stock_prices['returns'].fillna(0, inplace=True)

def map_sentiment(news, ticker_mapping):
    """
    Maps ticker_id with different forms of ticker_names

    Parameters
    ---------
    news : DataFrame row
        row of data about stock news
    ticker_mapping : map
        maps ticker_id with different forms of ticker_names
    """
    ticker_id = news['ticker_id']
    ticker_names = news['ticker_name']
    value = []
    for ticker_name in ticker_names:
        value.append(ticker_name)
    ticker_mapping[ticker_id] = value

def calculate_total_news(news, news_type, ticker_mapping):
    """
    Computes total positive and negative news for respective ticker_id

    Parameters
    ---------
    news : DataFrame row
        row of data about stock news
    news_type : str
        positive or negative
    ticker_mapping : map
       mapping of ticker_id with different forms of ticker_names

    Returns
    ------
    count : int
        count of positive or negative news based on news_type
    """
    news_for = news['news_for']
    ticker_ids = ticker_mapping[news_for]
    count = 0
    for sentiment in news['sentiment_analysis']:
        if sentiment ==None:
            continue
        else:
            if sentiment['sentiment_class']==news_type:
                for entity in sentiment['entities']:
                    if entity['entity'] in ticker_ids:
                        count+=1
                        break
    
    
            
    return count

def calculate_sentiment(news):
    """
    Compute sentiment score using formula (pos-neg)/(pos+neg)

    Parameters
    ---------
    news : DataFrame row
        row of data about stock news

    Returns
    ------
    sentiment score : float
        sentiment score for each row
    """
    if (news['positive']==0 and news['negative']==0):
        return 0
    return (news['positive']-news['negative'])/(news['positive']+news['negative'])

def generate_x(row, agg_index, stock_sentiment, agg_sentiment):
    """
    Create X vector with sentiment
    X = [returns, sector_return, sentiment_score, sector_sentiment_score, volume, sector_volume]

    Parameters
    ---------
    row : DataFrame row
        row of numerical data about the stock
    agg_index: DataFrame
        grouped numerical data about stocks
    stock_sentiment: DataFrame
        data about stock news
    agg_sentiment: DataFrame
        grouped data about stock news

    Returns
    ------
    X : array
        X for each row
    """
    ticker_id = row['ticker_id']
    date = row['date']
    returns = row['returns']
    sector_return = agg_index.loc[agg_index['date'] == date]['returns'].values[0]
    try:
        stock_sentiment_score = stock_sentiment.loc[(stock_sentiment['news_for'] == ticker_id) & (stock_sentiment['news_date'] == date)]['sentiment'].values[0]
    except IndexError:
        stock_sentiment_score = 0
    try:
        sector_sentiment_score = agg_sentiment.loc[agg_sentiment['news_date'] == date]['sentiment'].values[0]
    except IndexError:
        sector_sentiment_score = 0
    volume = row['volume']
    sector_volume = agg_index.loc[agg_index['date'] == date]['volume'].values[0]
    return [returns, 
            sector_return,
            stock_sentiment_score,
            sector_sentiment_score,
            volume,
            sector_volume]

def generate_x_ws(row, agg_index):
    """
    Create X vector without sentiment
    X = [returns, sector_return, volume, sector_volume]

    Parameters
    ---------
    row : DataFrame row
        row of numerical data about the stock
    agg_index: DataFrame
        grouped numerical data about stocks
    
    Returns
    ------
    X : array
        X for each row
    """
    ticker_id = row['ticker_id']
    date = row['date']
    returns = row['returns']
    sector_return = agg_index.loc[agg_index['date'] == date]['returns'].values[0]
    volume = row['volume']
    sector_volume = agg_index.loc[agg_index['date'] == date]['volume'].values[0]
    return [returns, 
            sector_return,
            volume,
            sector_volume]

def build_feature(data, size_of_x):
    """
    Create feature vector = [X(t-2) X(t-1) X(t)]

    Parameters
    ---------
    data : DataFrame
        numerical processed data about the stocks
    size_of_x: int
        size of x : with sentiment = 6 
                    without sentiment = 4
    """
    data['feature'] = data.apply(lambda row : np.zeros(shape=(3,size_of_x)), axis = 1)
    feature = [None, None]
    for i in range(2,len(data)):
        v = []
        if (data.loc[i-2,'ticker_id']==data.loc[i,'ticker_id']):
            v.append(data.loc[i-2, 'x'])
        else: 
            feature.append(None)
            continue
        if (data.loc[i-1,'ticker_id']==data.loc[i,'ticker_id']):
            v.append(data.loc[i-1, 'x'])
        else: 
            feature.append(None)
            continue
        v.append(data.loc[i, 'x'])
        v = normalize(v, axis=0).ravel()        
        feature.append(v)
    data['feature'] = feature
    data.pop('x')

def binary_classify(output):
    """
    Convert output to binary
    
    Parameters
    ---------
    output : float
        float output predicted for each stock
    
    Returns
    ------
    binary : int
        0/1 based on float value
    """
    if output>=0:
        return 1
    else:
        return 0

def process_output(output):
    """
    Convert binary to keywords

    Parameters
    ---------
    output : int
        binary output predicted for each stock
    
    Returns
    ------
    prediction : str
        RISE/DROP based on 1/0
    """
    if output==1:
        return 'RISE'
    elif output==0:
        return "DROP"
    else: 
        return np.NaN

def get_classifier():
    """
    Builds RandomForest classifier object

    Returns
    ------
    randomForestClassifier : RandomForestClassifier
        RandomForest Classifier object
    """
    return RandomForestClassifier(
        n_estimators=50,
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=0,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
    )

def get_cross_validator(folds):
    """
    Builds KFold object with desired folds

    Parameters
    ---------
    folds : int
        number of desired folds for the validator

    Returns
    ------
    kFold : KFold
        KFold object with desired folds
    """
    return KFold(n_splits=folds, shuffle=False)

def compute_auc(classifier, X, y):
    """
    Compute auc score

    Parameters
    ---------
    classifier : RandomForestClassifier
        classifier object
    X: array
        feature vector
    y: array
        actual output

    Returns
    ------
    auc_score : float
        auc_score for X and y
    """
    y_pred = classifier.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    auc_score = auc(fpr, tpr)
    return round(auc_score,3)

def compute_f1(classifier, X, y):
    """
    Compute f1 score

    Parameters
    ---------
    classifier : RandomForestClassifier
        classifier object
    X: array
        feature vector
    y: array
        actual output

    Returns
    ------
    f1_score : float
        f1_score for X and y
    """
    y_pred = classifier.predict(X)
    return round(f1_score(y,y_pred,average='macro'),3)

def compute_accuracy_score(classifier, X, y):

    """
    Compute accuracy score

    Parameters
    ---------
    classifier : RandomForestClassifier
        classifier object
    X: array
        feature vector
    y: array
        actual output

    Returns
    ------
    accuracy_score : float
        accuracy_score for X and y
    """
    y_pred = classifier.predict(X)
    return round(accuracy_score(y,y_pred),3)

def save_model_to_db(model , object_name):
    """
    Save trained model to database

    Parameters
    ---------
    model: RandomForestClassifier
        classifier object
    object_name: Name of file with which the model is to be saved in S3 bucket
    Returns
    ------
    None
    """
    #pickling the model
    #pickle.dumps(model , open('prediction_model.pkl' , 'wb'))
    pickled_model = pickle.dumps(model)
    #saving model to aws S3
    s3_client = boto3.client('s3',region_name='ap-south-1')

    response=s3_client.put_object(Bucket=  'fabric-prediction-models' , Key= object_name , Body= pickled_model )
    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print('\n**********************S3 bucket updated with model******************\n')
    else:
        print('Model could not be saved to S3 bucket response code from s3 :', response['ResponseMetadata']['HTTPStatusCode'])

def build_model(data,  folds = 5):
    """
    Build model and validate

    Parameters
    ---------
    data: DataFrame
        numerical processed data about the stocks
    folds: int 
        number of folds to use in KFold

    Returns
    ------
    clf : RandomForestClassifier
        trained classifier object
    """
    validation = pd.DataFrame()
    clf = get_classifier()
    cv = get_cross_validator(folds)
    accuracy_score = []
    scores = []
    for (train_index, test_index), i in zip(cv.split(data), range(folds)):
        training_data, testing_data = data.iloc[train_index], data.iloc[test_index]     
        clf.fit(list(training_data.feature), list(training_data.output))
        y_pred = clf.predict(list(testing_data.feature))
        # accuracy_score.append(accuracy_score( list(testing_data.output), y_pred))
        testing_data_with_output = testing_data.copy()
        testing_data_with_output['prediction'] = y_pred
        validation = validation.append(testing_data_with_output, ignore_index=True)

        
    return clf

def build_function_sector(stock_prices,stock_live_prices,stock_mapping,stock_news,MODELS_COL,sector , number_of_days_for_model_training , s3_bucket_object_name):
    """
    Process data

    Parameters
    ---------
    stock_prices: DataFrame
        numerical closing data about the stocks
    stock_live_prices: DataFrame 
        numerical live data about the stocks
    input_sentiment_csv: str
        path to sentiment csv
    STOCK_NEWS_COL: 
        column where the stocks news data is to be fetched from
    MODELS_COL:
        column where the model is to be saved or fetched from
    sector: str
        type of sector

    Returns
    ------
    output : array
        [output (dataframe), past_prediction_df (dataframe)]
    """
    global date_to_predict, data_train,all_stocks,all_stocks_bank
    stock_prices['date'] = pd.to_datetime(stock_prices['date']).dt.date
    stock_live_prices['date'] = pd.to_datetime(stock_live_prices['date']).dt.date

    count_per_ticker = stock_prices.groupby(['ticker_id'], as_index = False).count()
    total_stocks = len(count_per_ticker)
  
    all_stocks = list(count_per_ticker.ticker_id)


    build_returns_and_output(stock_prices)
    build_returns_and_output(stock_live_prices)
    #print("stock_live_prices after building returns\n" , stock_live_prices.tail(12))

    agg_index = stock_prices.groupby('date', as_index = False).agg({'returns':'mean', 'volume':'sum'}) #$$$ on all stocks??
    agg_live_index = stock_live_prices.groupby('date', as_index = False).agg({'returns':'mean', 'volume':'sum'})
    
    ticker_mapping= {} 
    stock_mapping.apply(lambda row : map_sentiment(row, ticker_mapping), axis = 1)
    
    stock_news['news_for'] = stock_news['news_for'].replace(stock_mapping.news_for.values,stock_mapping.ticker_id.values)

    stock_news['news_date'] = pd.to_datetime(stock_news['news_date']).dt.date

    stock_news['positive'] = stock_news.apply(lambda row : calculate_total_news(row, 'positive',ticker_mapping), axis = 1)
    stock_news['negative'] = stock_news.apply(lambda row : calculate_total_news(row, 'negative',ticker_mapping), axis = 1)
    stock_news['sentiment'] = stock_news.apply(lambda row : calculate_sentiment(row), axis = 1)

    stock_sentiment = stock_news.groupby(['news_for','news_date'], as_index = False).agg({'positive': "sum", 'negative': "sum"})
    stock_sentiment['sentiment'] = stock_sentiment.apply(lambda row : calculate_sentiment(row), axis = 1)

    agg_sentiment = stock_news.groupby('news_date', as_index = False).agg({'positive': "sum", 'negative': "sum"})
    agg_sentiment['sentiment'] = agg_sentiment.apply(lambda row : calculate_sentiment(row), axis = 1)

    stock_prices_ws = stock_prices.copy()
    stock_live_prices_ws = stock_live_prices.copy()

    stock_prices['x'] = stock_prices.apply(lambda row : generate_x(row,agg_index,stock_sentiment,agg_sentiment), axis = 1)
    stock_live_prices['x'] = stock_live_prices.apply(lambda row : generate_x(row,agg_live_index,stock_sentiment,agg_sentiment), axis = 1)


    stock_prices_ws['x'] = stock_prices_ws.apply(lambda row : generate_x_ws(row,agg_index), axis = 1)
    stock_live_prices_ws['x'] = stock_live_prices_ws.apply(lambda row : generate_x_ws(row,agg_live_index), axis = 1)

    def sortANDfilter_stock_data(stock_data):
        data=stock_data.sort_values(['ticker_id','date'])
        data = data[['ticker_id','date','x','output']]
        data.reset_index(drop = True, inplace = True)
        return data

    data= sortANDfilter_stock_data(stock_prices)
    data_ws= sortANDfilter_stock_data(stock_prices_ws)
    live_data= sortANDfilter_stock_data(stock_live_prices)
    live_data_ws= sortANDfilter_stock_data(stock_live_prices_ws)
    print("\n**************** data sorted by ticker id, date and only columns named ticker_id , date , x , output are considered***********\n")

    build_feature(data, 6)
    data.dropna(inplace=True)
    data.reset_index(drop = True, inplace = True)
    
    build_feature(data_ws, 4)
    data_ws.dropna(inplace=True)
    data_ws.reset_index(drop = True, inplace = True)

    build_feature(live_data, 6)
    live_data.sort_values(['date','ticker_id'], inplace = True)
    past_live_data_test = live_data[-2*total_stocks:-total_stocks]
    past_live_data_test.reset_index(drop = True, inplace = True)
    live_data = live_data[-total_stocks:]
    live_data.pop('output')
    live_data.reset_index(drop = True, inplace = True)
    try:date_to_predict = past_live_data_test['date'][0] ## changed
    except: pass

    build_feature(live_data_ws, 4)
    live_data_ws.sort_values(['date','ticker_id'], inplace = True)
    past_live_data_test_ws = live_data_ws[-2*total_stocks:-total_stocks]
    past_live_data_test_ws.reset_index(drop = True, inplace = True)
    live_data_ws = live_data_ws[-total_stocks:]
    live_data_ws.pop('output')
    live_data_ws.reset_index(drop = True, inplace = True)


    data['output'] = data['output'].apply(lambda x : binary_classify(x))
    data = data[['ticker_id', 'date', 'feature', 'output']]

    data_ws['output'] = data_ws['output'].apply(lambda x : binary_classify(x))
    data_ws = data_ws[['ticker_id', 'date', 'feature', 'output']]


    # data_train, data_val = train_test_split(data, test_size=0.3, shuffle = True)
    data.sort_values(['date','ticker_id'], inplace = True)
    model_training_duration = number_of_days_for_model_training
    data_train = data[-model_training_duration*total_stocks:]
    past_close_data_test = data[-total_stocks:]
    #past_close_data_test = data[-2*total_stocks:-total_stocks] ## changed
    data_train.reset_index(drop = True, inplace = True)
    past_close_data_test.reset_index(drop = True, inplace = True)


    data_ws.sort_values(['date','ticker_id'], inplace = True)
    data_train_ws = data_ws[-model_training_duration*total_stocks:]
    past_close_data_test_ws = data_ws[-total_stocks:] ## changed
    data_train_ws.reset_index(drop = True, inplace = True)
    past_close_data_test_ws.reset_index(drop = True, inplace = True)

    model = build_model(data_train)
    model_ws = build_model(data_train_ws)

    output = model.predict(list(live_data.feature))
    scores = np.max(model.predict_proba(list(live_data.feature)), axis=1)
    live_data['output'] = output
    live_data['output'] = live_data['output'].apply(lambda x : process_output(x))
    live_data['confidence_score'] = scores

    output_ws = model_ws.predict(list(live_data_ws.feature))
    scores_ws = np.max(model_ws.predict_proba(list(live_data_ws.feature)), axis=1)
    live_data_ws['output_without_sentiment'] = output_ws
    live_data_ws['output_without_sentiment'] = live_data_ws['output_without_sentiment'].apply(lambda x : process_output(x))
    live_data_ws['confidence_score_ws'] = scores_ws
    try: prediction_date = live_data['date'][0]
    except: print("No data in live_data variable")
    
    ## passing the model to save in s3 bucket
    save_model_to_db(model , s3_bucket_object_name)

    output= pd.merge(live_data, live_data_ws, on='ticker_id')
    output = output[['ticker_id','output','confidence_score','output_without_sentiment','confidence_score_ws']]

    output['actual_output']= output['ticker_id'].apply(lambda x: binary_classify(list(past_live_data_test[past_live_data_test.ticker_id == x].output)[0]))
    output['close_expected_output'] =output['ticker_id'].apply(lambda x: model.predict(list(past_close_data_test[past_close_data_test.ticker_id == x].feature)) [0])
    output['close_expected_output_without_sentiment'] =output['ticker_id'].apply(lambda x: model_ws.predict(list(past_close_data_test_ws[past_close_data_test_ws.ticker_id == x].feature)) [0])
    output['actual_output'] = output['actual_output'].apply(lambda x : process_output(x))
    output['close_expected_output']=output['close_expected_output'].apply(lambda x : process_output(x))
    output['close_expected_output_without_sentiment']=output['close_expected_output_without_sentiment'].apply(lambda x : process_output(x))
    

    return output

#__________________________________________________________________________________________________________________________

##########################################################################################################################
'''Starting point of execution for above fucntions'''
#########################################################################################################################

if(it_latest_date == TODAY_DATE):
    it_output = build_function_sector(ITstock_close_price,ITstock_live_price,stock_mapping = build_sentiment_mapping_csv(algoDDB , 'IT'), stock_news = IT_News, MODELS_COL = "path_to_model_storage", sector='it', number_of_days_for_model_training = model_training_duration ,s3_bucket_object_name='IT_model')
    print(it_output)
else: print("\nLatest IT stock data is not captured\n")

if(bank_latest_date == TODAY_DATE):
    bank_output = build_function_sector(Bankstock_close_price,Bankstock_live_price,stock_mapping =  build_sentiment_mapping_csv(algoDDB , 'BANK'), stock_news = Bank_News, MODELS_COL = "path_to_stored_model", sector='bank', number_of_days_for_model_training = model_training_duration, s3_bucket_object_name='BANK_model')
    print(bank_output)
else:print("\nLatest bank stock data is not captured\n")

## combining IT and BANK output dataframes
if(len(it_output)>0 or len(bank_output)>0): 
    output= pd.concat([it_output ,bank_output])
    print(output)
else:
    print("\nBoth dataframes (it_output and bank_output) are empty\n")
    sys.exit()



###########################################################################################################################
''' Writing data to RDS'''
#############################################################################################################################
def write_data_to_RDS(output):
    db=get_connection(host , user , token , port , database)
    cursor=db.cursor()
    for i , row in output.iterrows():
        query= "Insert into "+rds_stock_prediction_table+" VALUES (%s, %s,%s, %s,%s,%s ,%s, %s ,%s )"
        cursor.execute(query ,(row['ticker_id'] , 
        TODAY_DATE, 
        row['output'], 
        row['confidence_score'] ,
        row['confidence_score_ws'] , 
        row['output_without_sentiment']	,
        row['close_expected_output_without_sentiment'],
        row['close_expected_output'],
        row['actual_output']))
    cursor.execute("commit")
    cursor.close()
    db.close()
    print("Prediction data written into Table")

## calling above function to write data to RDS
write_data_to_RDS(output)


