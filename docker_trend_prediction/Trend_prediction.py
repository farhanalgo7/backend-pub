#imports
from unicodedata import name
import pandas as pd
import numpy as np
from math import *
from pip import main
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from datetime import date
from scipy import stats
#
import boto3
import warnings
import json
import  sys
import mysql.connector
from datetime import datetime , timezone
import datetime as dt
import os
os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'

#-------------------------------------------Functions Declaration--------------------------------------------------------

def compute_trend_variable(group):
    """
    Calculates features i.e trend variables for 2,3,5,10 days

    Parameters
    ---------
    group: DataFrame group
        group of ticker_id
    
    Returns
    ------
    df: Dataframe
        dataframe for all features for ticker_ids

    Steps
    -----
    1. Make input array of closing prices
    2. Iterate over number of days(k) to calculate trend variable
        2.1 Append NaN values for first 10 days to temporary array td
        2.2 Iterate from 10th day to avoid undefined values of t10
            2.2.1 Calculate sum of absolute differences of closing prices of consecutive days for past k days
            2.2.2 Divide the absolute difference of closing price of ith day and (i-k)th day by sum and append final value to td araay 
        
    """

    #array of number of days to calculate trend variables
    d = [2,3,5,10]  

    #empty array to store trend variable data
    t = []          

    #Step 1: Input array containing closing price
    cp_data = group['close'] 
    cp_data = cp_data.reset_index(drop=True)

    #Step 2: Iterate over number of days to calculate trend variable
    for k in d:

        #create empty array to store trend variable arrays
        td=[]

        #Step 2.1: Append NaN values for first 10 days
        for i in range(10):
            td.append(np.nan)

        #Step 2.2: Iterate from 10th day to avoid undefined values of t10
        for i in range (10, len(cp_data)):
            #Step 2.2.1: Calculate sum of absolute differences of closing prices of consecutive days for past k days
            sum = 0
            for j in range (i-k, i):
                sum+= abs(cp_data[j+1] - cp_data[j])
            #Step 2.2.2: Divide the absolute difference of closing price of ith day and (i-k)th day by sum and append final value to td araay
            td.append(abs(cp_data[i]-cp_data[i-k])/sum)

        #append the array containing trend variable over past k days to main array
        t.append(td)

    #return dataframe of t2,t3,t5 and t10
    return pd.DataFrame({"t2":t[0], "t3":t[1],"t5":t[2],"t10":t[3] }, index=group.index)

def compute_output(group, no_of_days):  
    """
    Calculates output for ticker_ids

    Parameters
    ---------
    group: DataFrame group
        group of ticker_id
   
    no_of_days: Integer
        Integer specifying the no of days for prediction
    
    Returns
    ------
    df: Dataframe
        dataframe for output for ticker_ids

    Steps
    -----
    1. Create output array
    2. Append output values
    3. Return output dataframe

    """
    
    #copy the values in a temporary variable
    temporary = group.copy()
    temporary.reset_index(drop = True, inplace = True)

    #Step 1: Create empty array to store output
    output = []

    #get the trend variable for output using no_of days
    trend_variable = "t" + str(no_of_days)

    #Step 2:Append output values
    #iterate over temporary array to append output values
    for i in range(len(temporary)- no_of_days):
        #append the trend variable value of no_of_days(5) ahead in output variable
        output.append(temporary[trend_variable][i + no_of_days])

    #append NaN values for last no_of_days values
    output.extend([np.nan]*5)

    #Step 3: Return output dataframe
    return pd.DataFrame({'output':output}, index=group.index)

def get_regressor():
    """
    Builds RandomForest regressor object

    Returns
    ------
    randomForestRegressor: RandomForestRegressor
        RandomForest Regressor object
    """

    return RandomForestRegressor(n_estimators = 100, random_state = 0)

def build_model(data):
    """
    Build model

    Parameters
    ---------
    data: DataFrame
        numerical processed data about the stocks

    Returns
    ------
    clf: RandomForestRegressor
        trained classifier object
    """
    reg = get_regressor()
    reg.fit(list(data.feature), list(data.output))
    return reg

def train_model(group, results, window_size = 60):
    """
    Trains the model using window method

    Parameters
    ---------
    group: DataFrame group
        group of ticker_id
    results: DataFrame
        consists actual and predicted data
    window_size: Integer
        length of window
        
    Returns
    ------
    results: DataFrame
        consists actual and predicted data

    Steps
    -----
    1. Training
        1.1 Create train and test data using window_size
        1.2 Build model with train data
        1.3 Predict for test data
        1.4 Calculate percentile for test data
        1.5 Create result object and convert into dataframe
        1.6 Append results to dataframe
    """

    data = group.copy()
    # print(group.name)
    data.reset_index(drop = True, inplace = True)

    #sort data according to date
    data.sort_values(['date'], inplace = True)

    #Step 1: Training
    for day in range(len(data)-window_size-5):
        #print('Ticker: %s, Window: %s' % (group.name, day))
        #Step 1.1 Create train and test data using window_size
        data_train = data[day:day+window_size]
        data_test = data.iloc[day+window_size+5]
        prediction_date = data_test.date

        #Step 1.2 Build model with train data
        model = build_model(data_train)

        #Step 1.3 Predict for test data
        y = model.predict([list(data_test.feature)])

        #Step 1.4 Calculate percentile for test data
        percentile_y = stats.percentileofscore(data_train['output'], y[0])
        
        
        #Step 1.5 Create result object and convert into dataframe
        row = {
            'date':prediction_date,
            'ticker_id':group.name,
            'output':data_test.output,
            'prediction':y,
            'percentile': percentile_y
       }
        df = pd.DataFrame(row)

        #Step 1.6 Append results to dataframe
        results = results.append(df,ignore_index = True)
        
    #return results
    return results

#_________________________________________________________________________________________________________

if __name__=="__main__":

    TODAY = dt.datetime.now(timezone.utc)+dt.timedelta(hours=5, minutes = 30)
    today_date = TODAY.strftime('%Y-%m-%d') ## changing date format
    #input file
    #stock_prices = pd.read_csv('./Input/StockClosePrices (1).csv')
    #stock_prices = pd.read_csv('./StockClosePrices.csv')

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
    trends_table=credentials['trends_prediction_stock_table']
    #____________________________________________________________________________________________________________________________________________

    #----------------------------------- SECTION - connection to AWS RDS------------------------------------------------------------

    client= boto3.client('rds', region_name=region)
    ## generating a database authentication token for user mentioned in IAM policy of AWS
    try: token = client.generate_db_auth_token(DBHostname=host, Port=str(port), DBUsername=user, Region=region)
    except Exception as err: 
        print("\n could not get authentication token for database, getting error :",err)
        sys.exit()

    def get_connection():
        return mysql.connector.connect(host=host , user=user , password=token , port=port , database=database)

    ## calling get_connection method to get connection object
    while(1):
        try: 
            db= get_connection()
            break
        except: 
            print("\ncould not establish connection with RDS..Retrying..................")
            continue
    print("************************Connection established with AWS RDS***************")
    #______________________________________________________________________________________________________________________


    #---------------------------------------SECTION- Data fetch from RDS-----------------------------------------

    TODAY = dt.datetime.now(timezone.utc)+dt.timedelta(hours=5, minutes = 30)
    today_date= TODAY.strftime('%Y-%m-%d') 
    # date 60 days back
    d=(datetime.now(timezone.utc) - dt.timedelta(60)+ dt.timedelta(hours=5 ,minutes=30)).strftime('%Y-%m-%d')
    stock_prices = pd.read_sql("select * from "+rds_stock_data_table+" where HOUR(DATE) > 14 and date > "+d+" order by date",db)
    print("*********************Fetched stock close prices of last 60 days***************************")
    #___________________________________________________________________________________________________________


    #drop unnecessary columns and convert date to required format
    stock_prices = stock_prices.drop(['adj_close', 'high', 'low', 'open', 'volume'], axis=1)
    stock_prices['date'] = pd.to_datetime(stock_prices['date']).dt.date

    #store total number of stocks and their names
    count_per_ticker = stock_prices.groupby(['ticker_id'], as_index = False).count()
    total_stocks = len(count_per_ticker)

    #sort data according to ticker_id and date
    stock_prices = stock_prices.sort_values(['ticker_id','date'])
    stock_prices.reset_index(drop = True, inplace = True)

    #define list of features
    features = ['t2','t3','t5','t10']

    #compute trend variables
    stock_prices[features] = stock_prices.groupby(['ticker_id']).apply(lambda group : compute_trend_variable(group))
    stock_prices = stock_prices.dropna(subset=['t2', 't3', 't5', 't10'])

    #compute output
    stock_prices[['output']] = stock_prices.groupby(['ticker_id']).apply(lambda group : compute_output(group, 5))

    #generate feature array
    stock_prices['feature'] = stock_prices[features].values.tolist()
    total_features = len(features)

    #Results
    ## create results dataframe
    results = pd.DataFrame(columns=['date','ticker_id','output','prediction', 'percentile'])
    ## get training results
    results = stock_prices.groupby(['ticker_id']).apply(lambda group : train_model(group, results))
    results = results.sort_values(['date'])
    results.reset_index(drop = True, inplace = True)

    #store last 10 days' results
    predictions = results.tail(10*total_stocks)
    predictions = predictions.sort_values(['ticker_id','date'])
    predictions = predictions.reset_index(drop = True)
    #predictions.to_csv("Predictions "+ str(date.today())+".csv")
    
    Predictions = predictions.groupby(['ticker_id'], as_index=False).aggregate({"prediction":lambda x: ",".join(map(str, round(x,4) )) ,
    "date":lambda x: ",".join(map(str, x)), "output":lambda x: ",".join(map(str, round(x,4) )) , "percentile":lambda x: ",".join(map(str, round(x,4) ))})
    
    print(Predictions)
    #-----------------------------------------Storing predictions to RDS----------------------------------------------------
    def save_prediction_to_RDS(row):
        query= "INSERT INTO "+trends_table+" VALUES (%s,%s,%s,%s,%s,%s)"
        cursor.execute(query , (today_date,row['date'],row['ticker_id'],row['output'],row['prediction'],row['percentile']) )
        cursor.execute("commit")
    cursor=db.cursor()
    print("*******************started writing data to RDS********************************")
    for i,row in Predictions.iterrows():
        save_prediction_to_RDS(row)
    print("********************Trend Prediction data written to RDS table"+trends_table+"******************************")

    cursor.close()
    db.close()
    
    #_______________________________________________________________________________________________________________

    


