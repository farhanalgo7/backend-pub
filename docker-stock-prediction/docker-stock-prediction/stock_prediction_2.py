"""  Module to do predictions on stocks from various sectors and then insert the prediction table into AWS RDS

1. Get stock closing price from RDS
2. Compute indicators
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
    Ekansh Gupta <egupta@algoanalytics.com>
    Divyank Lunkad <dlunkad@algoanalytics.com>
    Kalash Gandhi <kgandhi@algoanalytics.com>
    Mrityunjay Samanta <msamanta@algoanalytics.com>

Created: 5 January 2022

"""
import pandas as pd
import numpy as np
from math import *
import mysql.connector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score
import pymysql
from talib.abstract import *
from talib import MA_Type
from sklearn.metrics import plot_confusion_matrix, roc_curve, accuracy_score, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay
import boto3
from boto3.dynamodb.conditions import Key , Attr
import sys
import json
from datetime import datetime , timezone
import datetime as dt
import os
#os.system('python3 stock_prediction_1.py')

os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'

#-----------------------------------SECTION - Fetching global variables from Secret manager----------------------------------------------
"""  About function get_secret
Function to get secret manager connection object
Input : None
Output: Secret manager connection object
"""
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
## getting secret string containing variables for RDS
credentials = json.loads(get_secret()['SecretString'])
## initializing variables with the help of secret manager keys
host = credentials['host']
user= credentials['username']
port= credentials['port']
database= credentials['Database_Name']
region= credentials['region_name']
rds_stock_data_table = credentials['Ingestion_table_name']
rds_stock_next5Days_prediction_table= credentials['rds_stock_next5Days_prediction_table']
rds_stock_last5Days_prediction_table= credentials['rds_stock_last5Days_prediction_table']

## initializing today's date in IST
TODAY = dt.datetime.now(timezone.utc)+dt.timedelta(hours=5, minutes = 30)
today_date = TODAY.strftime('%Y-%m-%d') ## changing date format
#____________________________________________________________________________________________________________________________________________


#----------------------------------- SECTION - connection to AWS RDS------------------------------------------------------------

# initializing boto client
client= boto3.client('rds', region_name=region)
## generating a database authentication token for user mentioned in IAM policy of AWS
try: token = client.generate_db_auth_token(DBHostname=host, Port=port, DBUsername=user, Region=region)
except Exception as err: 
    print("\n could not get authentication token for database, getting error :",err)
    sys.exit()

"""  About function get_connection
Function to get connection object for AWS RDS
Input :
      host: host address
      user: username
      database: database name which is to be connected
      password: password to connect to database
      port: port number
Output: 
      returns mysql connector object
"""
def get_connection(host , user , token , port , database):
  return mysql.connector.connect(host=host , user=user , password=token , port=port , database=database)

count = 10
## calling get_connection method inside loop to rety attemps for RDS connection in case of any errors/exception
while(count>0):
    try: 
        db= get_connection( host , user , token , port , database)
        print("Connection Established!! Woohoo!!")
        break
    except: 
        print("\ncould not establish connection with RDS..Retrying.................... Retry Countdown : ",count)
        count-=1
        continue

#________________________________________________________________________________________________________________________________________________

#---------------------------------------SECTION- connection to dynamodb--------------------------------------------------------------------------
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

## initializing dynamoDB  boto resource object
algoDDB= boto3.resource('dynamodb',region_name=region)
## initializing ConfigTable
table=algoDDB.Table('ConfigTable')

## get ticker_ids and ticker_names for Bank and IT sector
IT_ticker_id , IT_ticker_name = get_ticker_names_and_id('IT', table)
BANK_ticker_id , BANK_ticker_name = get_ticker_names_and_id('BANK' , table)

#_______________________________________________________________________________________________________________________--

#---------------------------------------SECTION- connection to AWS RDS--------------------------------------------------------------------------

## getting mysql connection object
db= get_connection(host , user , token , port , database)

## creating placeholder to used at time of quering stock table using pandas read_sql function
bank_placeholder="%s" + ''.join(',%s' * (len(BANK_ticker_id)-1))
it_placeholder="%s" + ''.join(',%s' * (len(IT_ticker_id)-1))

# geting close price for IT and Bank sector
print("*************** started fetching close stock data for IT and BANK sector******************")
Bankstock_close_price = pd.read_sql("select * from "+rds_stock_data_table+" where ticker_id IN ("+bank_placeholder+") AND HOUR(DATE) > 14 order by date",db,params=BANK_ticker_id)
ITstock_close_price = pd.read_sql( "select * from "+rds_stock_data_table+" where ticker_id IN ("+it_placeholder+") AND HOUR(DATE) > 14 order by date",db,params=IT_ticker_id)

print("*************** fetched stock close data for IT and BANK sector******************")

#____________________________________________________________________________________________________________________________________________

#---------------------------------------SECTION- Functions declaration--------------------------------------------------------------------------

##indicators which we will be using as features
indicators = ['simple_moving_average',
              'exponential_moving_average',
              'relative_strength_index',
              'standard_deviation',
              'average_directional_index',
              'stochastic_oscillator',
              'moving_average_convergence_divergence',
              'bollinger_bands'
             ]

def compute_indicators(group):
    """
    Calculates all indicator for ticker_ids

    Parameters
    ---------
    group : DataFrame group
        group of ticker_id
    
    Returns
    ------
    df : Dataframe
        dataframe for all indicators for ticker_ids
    """
    #Simple moving average
    sma = []
    _sma = SMA(group['close']
               , timeperiod=14)
    for a in _sma:
        sma.append([a])
    
    #Exponential moving average
    ema = []
    _ema = EMA(group['close'], timeperiod=14)
    for a in _ema:
        ema.append([a])
    
    #Relative strength index
    rsi = []
    _rsi = RSI(group['close'], timeperiod=14)
    for a in _rsi:
        rsi.append([a])
    
    #Standard deviation
    sd = []
    _sd = STDDEV(group['close'], timeperiod=5, nbdev=1.0)
    for a in _sd:
        sd.append([a])
        
    #Average directional index
    adi = []
    _adi = ADX(group['high'], group['low'], group['close'], timeperiod=14)
    for a in _adi:
        adi.append([a])
        
    #Stochastic oscillator
    stoch = []
    _fastk, _fastd = STOCHF(group['high'], group['low'], group['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    for (a,b) in zip(_fastk, _fastd):
        stoch.append([a,b])
    
    #Moving average convergence divergence
    macd = []
    _macd, _macdsignal, _macdhist = MACD(group['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    for (a,b,c) in zip(_macd, _macdsignal, _macdhist):
        macd.append([a,b,c])
        
    #Bollinger bands
    bb = []
    _upper, _middle, _lower = BBANDS(group['close'], matype=MA_Type.T3)
    for (a,b,c) in zip(_upper, _middle, _lower):
        bb.append([a,b,c])
    
    return pd.DataFrame({'simple_moving_average':sma,
                         'exponential_moving_average':ema,
                         'relative_strength_index':rsi,
                         'standard_deviation':sd,
                         'average_directional_index':adi,
                         'stochastic_oscillator':stoch,
                         'moving_average_convergence_divergence':macd,
                         'bollinger_bands':bb
                        }, index=group.index)

def compute_output(group):  
    """
    Calculates output for ticker_ids

    Parameters
    ---------
    group : DataFrame group
        group of ticker_id
    
    Returns
    ------
    df : Dataframe
        dataframe for output for ticker_ids
    """
    temporary = group.copy()
    temporary.reset_index(drop = True, inplace = True)
    output = []
    for i in range(len(temporary)-5):
        more = 0
        less = 0
        for j in range(5):
            if (temporary['close'][i]>temporary['close'][i+j+1]):
                less += 1
            if (temporary['close'][i]<temporary['close'][i+j+1]):
                more += 1
        if more>=4:
            output.append(1)
        elif less>=4:
            output.append(-1)
        else:
            output.append(0)
    output.extend([np.nan]*5)
    return pd.DataFrame({'output':output}, index=group.index)

def generate_feature(row):
    """
    Builds feature vector

    Parameters
    ---------
    row : DataFrame row
        row of all indicator data about the stock

    Returns
    ------
    feature : array
        feature for each row
    """
    feature = []
    for i in indicators:
        for v in row[i]:
            if pd.isna(v):
                return np.NaN
            feature.append(v)
    return feature

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
    elif output==-1:
        return "DROP"
    else: 
        return "Can't Determine"

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

def build_model(data):
    """
    Build model

    Parameters
    ---------
    data: DataFrame
        numerical processed data about the stocks

    Returns
    ------
    clf : RandomForestClassifier
        trained classifier object
    """
    clf = get_classifier()
    clf.fit(list(data.feature), list(data.output))
    return clf

def train_model(group, results, window_size = 20):
    """
    Trains the model using window method

    Parameters
    ---------
    group : DataFrame group
        group of ticker_id
    results: DataFrame
        consists actual and predicted data
    window_size : Integer
        length of window
        
    Returns
    ------
    results : DataFrame
        consists actual and predicted data
    """
    data = group.copy()
    data.reset_index(drop = True, inplace = True)
    data.sort_values(['date'], inplace = True)
    print(data.tail())
    for day in range(len(data)-window_size-5):
        data_train = data[day:day+window_size]
        data_test = data.iloc[day+window_size+5]
        prediction_date = data_test.date
        model = build_model(data_train)
        y = model.predict([list(data_test.feature)])
 
        conf_score = np.max(model.predict_proba([list(data_test.feature)]), axis=1)
        row = {
            'date':prediction_date,
            'ticker_id':group.name,
            'output':data_test.output,
            'prediction':y,
            'confidence_score':conf_score
       }
        df = pd.DataFrame(row)
        results = results.append(df,ignore_index = True)
    return results

def main_function(stock_prices):
    """
    Function to build prediction dataframe

    Parameters:
        stock_prices: csv
            stock data
    
    Returns:
        prediction: csv
            prediction csv for last 5 days
    """

    stock_prices['date'] = pd.to_datetime(stock_prices['date']).dt.date
    count_per_ticker = stock_prices.groupby(['ticker_id'], as_index = False).count()
    total_stocks = len(count_per_ticker)
    all_stocks = list(count_per_ticker.ticker_id)
    stock_prices = stock_prices.sort_values(['ticker_id','date'])
    stock_prices.reset_index(drop = True, inplace = True)

    stock_prices[indicators] = stock_prices.groupby(['ticker_id']).apply(lambda group : compute_indicators(group))

    stock_prices[['output']] = stock_prices.groupby(['ticker_id']).apply(lambda group : compute_output(group))

    stock_prices['feature'] = stock_prices.apply(lambda row : generate_feature(row), axis = 1)
    stock_prices.dropna(subset = ['feature'], inplace=True)
    stock_prices.reset_index(drop = True, inplace = True)

    results = pd.DataFrame(columns=['date','ticker_id','output','prediction'])
    results = stock_prices.groupby(['ticker_id']).apply(lambda group : train_model(group, results))
    results = results.sort_values(['date'])
    results.reset_index(drop = True, inplace = True)

    prediction = results.tail(5*total_stocks)
    results.dropna(inplace=True)
    # #calling results function
    # display_training_results(results)
    last_to_last5days_prediction = results.tail(5*total_stocks)
    prediction.drop(['output'], axis=1, inplace=True)
    prediction.reset_index(drop = True, inplace = True)

    prediction = prediction.sort_values(['ticker_id','date'])
    prediction = prediction.reset_index(drop=True)
    prediction.head(10)

    past_prediction = last_to_last5days_prediction.sort_values(['ticker_id','date'])
    past_prediction = past_prediction.reset_index(drop=True)
    past_prediction.head(10)

    return prediction , past_prediction

#___________________________________________________________________________________________________________________________

#-----------------------Functions for accuracy , f1 score , AUC and all metrices-----------------------------------

def compute_auc(y, y_pred):
    """
    Compute auc score

    Parameters
    ---------
    y: array
        actual output
    y_pred: array
        predicted output

    Returns
    ------
    auc_score_neg : float
        auc_score for negative class 
    auc_score_neutral : float
        auc_score for neutral class 
    auc_score_pos : float
        auc_score for positive class 
    """
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=-1)
    label_neg = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=0)
    label_neutral = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1)
    label_pos = auc(fpr, tpr)
    return round(label_neg,3), round(label_neutral,3), round(label_pos,3)

def compute_f1(y, y_pred):
    """
    Compute f1 score

    Parameters
    ---------
    y: array
        actual output
    y_pred: array
        predicted output

    Returns
    ------
    f1_score : float
        f1_score for X and y
    """
    return round(f1_score(y,y_pred,average='macro'),3)

def compute_accuracy_score(y, y_pred):
    """
    Compute accuracy score

    Parameters
    ---------
    y: array
        actual output
    y_pred: array
        predicted output

    Returns
    ------
    accuracy_score : float
        accuracy_score for X and y
    """
    return accuracy_score(y, y_pred)

def display_training_results(results):
    """
    Displays the results for training the model

    Parameters
    ---------
    results : DataFrame
        consists actual and predicted data
    """
    auc_neg, auc_neutral, auc_pos = compute_auc(results.output,results.prediction)
    f1 = compute_f1(results.output,results.prediction)
    acc = compute_accuracy_score(results.output,results.prediction)
    print(f"Accuracy = {acc:.3f}  ||  AUC DROP = {auc_neg:.3f}  || AUC RISE = {auc_pos:.3f}  || F1 = {f1:.3f}")

#________________________________________________________________________________________________________________________


##########################################################################################################################
'''Starting point of execution for above fucntions'''
#########################################################################################################################

## calling main function on IT stock data
next_5_days_IT_prediction , past_IT_prediction=main_function(ITstock_close_price)
## calling main function on BANK stock data
next_5_days_Bank_prediction , past_Bank_prediction=main_function(Bankstock_close_price)

## checking if valid output given by main function for next_5_days_prediction
if(len(next_5_days_IT_prediction)>0 or len(next_5_days_Bank_prediction)>0): 
    ## combining IT and Bank dataframes
    prediction_all_sector= pd.concat([next_5_days_IT_prediction ,next_5_days_Bank_prediction])
    ## grouping dataframe values on ticker id
    prediction_all_sector = prediction_all_sector.groupby(['ticker_id'], as_index=False).aggregate({"prediction":lambda x: ",".join(map(str, x)) ,
    "date":lambda x: ",".join(map(str, x)) , "confidence_score":lambda x: ",".join(map(str, x))})
    print(prediction_all_sector.head())
else:
    print("\nBoth dataframes (next_5_days_IT_predictiont and next_5_days_Bank_prediction) are empty\n")
    sys.exit()

## checking if valid output given by main function for past_prediction
if(len(past_IT_prediction)>0 or len(past_Bank_prediction)>0): 
    ## combining IT and Bank dataframes
    past_prediction_all_sector= pd.concat([past_IT_prediction ,past_Bank_prediction])
    ## grouping dataframe values on ticker id
    past_prediction_all_sector = past_prediction_all_sector.groupby(['ticker_id'], as_index=False).aggregate({"prediction":lambda x: ",".join(map(str, x)) ,
    "date":lambda x: ",".join(map(str, x)),"output":lambda x: ",".join(map(str, x)), "confidence_score":lambda x: ",".join(map(str, x))
    })
    print(past_prediction_all_sector.head())
else:
    print("\nBoth dataframes (next_5_days_IT_predictiont and next_5_days_Bank_prediction) are empty\n")
    sys.exit()

############################################################################################################################
''' Writing data to RDS'''
#############################################################################################################################

"""  About function write data to RDS
Function to store data to AWS RDS
Input :
      output: Dataframe
      table: table in which data needs to be inserted
Output: 
      None
"""
def write_data_to_RDS(output , table):
    db=get_connection(host , user , token , port , database)# get connection object
    cursor=db.cursor()# initialize cursor
    if(table==rds_stock_next5Days_prediction_table):
        for i , row in output.iterrows():# iterate through each dataframe row
            # preparing query
            query= "Insert into "+table+" VALUES (%s, %s, %s, %s,%s)"
            ## execute query using cursor and entring all data
            cursor.execute(query ,(today_date, row['ticker_id'] , row['date'], row['prediction'],row['confidence_score']))
    if(table==rds_stock_last5Days_prediction_table):
        for i , row in output.iterrows():
            query= "Insert into "+table+" VALUES (%s, %s, %s, %s, %s,%s)"
            cursor.execute(query ,(today_date, row['ticker_id'] , row['date'], row['output'],row['prediction'],row['confidence_score']))

    cursor.execute("commit")# commiting to database
    cursor.close()#closing the cursor
    db.close()# closing db connection 
    

## calling above function to write data to RDS
write_data_to_RDS(prediction_all_sector , rds_stock_next5Days_prediction_table)
print("\n Next 5 days prediction for all sectors stored in RDS \n")
write_data_to_RDS(past_prediction_all_sector , rds_stock_last5Days_prediction_table)
print("\n Last 5 days prediction for all sectors stored in RDS \n")
