import boto3
from boto3.dynamodb.conditions import Key , Attr
from datetime import datetime , timezone
import pandas as pd
import numpy as np
import time
import preprocessor as p
import re
import string
from gensim import corpora,models
from gensim.models import LdaModel, CoherenceModel
import datetime as dt
from string import digits
#important libraries for preprocessing using NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from collections import Counter, defaultdict
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,strip_numeric
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
from multiprocessing import Process, freeze_support

nltk.download(["stopwords","wordnet",'omw-1.4' ])
DAYS=30 # Define number of past days we want to store JSON file for
TODAY = dt.datetime.now().strftime('%Y-%m-%d')
#Code for topic modeling

REGION = "ap-south-1"
BUCKET_NAME = "topics-prediction-bucket"
BUCKET = boto3.resource('s3')
algoDDB= boto3.resource('dynamodb',region_name= REGION, endpoint_url='http://dynamodb.ap-south-1.amazonaws.com')       



def clean_text_modified(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
 

    text = p.clean(text)
    text = text.lower()
    text = text.strip()
    
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text=re.sub('@\w+\b','',text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def text_preprocessing_fn(text):
    """
    Cleaning and parsing the text.
    """
    result_list = []
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    #Adding custom stopwords
    stopwords_custom = nltk.corpus.stopwords.words('english')

    for each_document in text:
      #Text preprocessing
      text_cleaned = clean_text_modified(each_document)
     

      tokens = tokenizer.tokenize(text_cleaned)

      remove_stopwords = [w for w in tokens if w not in stopwords_custom and len(w) > 2]

      #Lemmatization
      processed_text = " ".join(lemmatizer.lemmatize(token, pos='v') for token in remove_stopwords)
      combined_text = ''.join(processed_text)
      result_list.append(combined_text)
      
    return result_list
      
def Optimal_Model(tokenized_reviews):
  #Optimal Model
  dictionary = corpora.Dictionary(tokenized_reviews)
  doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]
  tfidf = models.TfidfModel(doc_term_matrix,smartirs='ntc',normalize=True)
  corpus_tfidf = tfidf[doc_term_matrix] 
  optimal_model=LdaModel(corpus=corpus_tfidf,id2word=dictionary,num_topics=5,update_every=1,chunksize=len(doc_term_matrix),passes=20,alpha='auto',random_state=123)
  model_topics = optimal_model.show_topics(formatted=False)
  coherence_model_lda = CoherenceModel(model=optimal_model, texts=tokenized_reviews, dictionary=dictionary , coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  
  return optimal_model,coherence_lda,corpus_tfidf

    
def format_topics_sentences(model,corpus,texts,date_time,id):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
    
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                
                wp = model.show_topic(topic_num)
                # getting keywords of that particular topics
                topic_keywords = ", ".join([word for word, prop in wp])
                
                #Creating dataframe
                sent_topics_df = sent_topics_df.append(pd.Series([topic_keywords]), ignore_index=True)
            else:
                break
    # naming dataframe columns 
    sent_topics_df.columns = ['Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([date_time,id,contents,sent_topics_df], axis=1)
    
    return(sent_topics_df)

def Word_Cloud_Freq(model,ideal_topic_num):
  topics=[]
  filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
  for topic in model.show_topics(num_topics=ideal_topic_num):
    topics.append(preprocess_string(topic[1], filters))
  Final_list=[]
  for i in range (len(topics)):
    Final_list +=topics[i]
  counts = Counter(Final_list)
  
  return topics,counts

def Topic_Modeling_Tweets(df, company):

    print(df)
    df['tweet_date'] = pd.to_datetime(df.tweet_date, format='%Y-%m-%d')
    df['tweet_date'] = df['tweet_date'].dt.strftime('%Y-%m-%d')
    dictionary_tweets={}
   

    for days in range(DAYS):

        start_time = dt.datetime.now().replace(microsecond=0)
        print(f"************************************ For day {days+1}**************************************************")

        try:   
          date = (dt.datetime.strptime(TODAY, '%Y-%m-%d') - dt.timedelta(days)).strftime('%Y-%m-%d')
          df_tweets= df[((df['tweet_date'] >= date )&(df['tweet_for']==company))]
          df_tweets=df_tweets.reset_index(drop=True)

          df_tweets['tweets']=df_tweets['tweet_text']
          text_list=df_tweets['tweet_text'] #.tolist()
          text_list = text_preprocessing_fn(text_list)
          tokenized_reviews=[x.split(" ") for x in text_list]
          s1 = dt.datetime.now().replace(microsecond=0)
          print(f'Tweets cleaned and time taken (text preprocessing/cleaning) {s1-start_time}')
          s4 = dt.datetime.now().replace(microsecond=0)
          optimal_model,coherence_lda,corpus_tfidf=Optimal_Model(tokenized_reviews)
          s7 = dt.datetime.now().replace(microsecond=0)
          print(f'Optimal_Model and time taken is {s7-s4}')
          Tweets_documents=format_topics_sentences(optimal_model,corpus_tfidf,df_tweets['tweets'],df_tweets['tweet_date'], df_tweets['tweets_ID'])
          # data_struct = lambda:defaultdict(list)
          temp_dict = defaultdict(list)
          for index,rows in  Tweets_documents.iterrows():
            #temp_dict[rows['Topic_Keywords']][rows['tweet_date'].split()[0]].append(rows['tweets_ID'])
            temp_dict[rows['Topic_Keywords']].append(rows['tweets_ID'])

          print("temp_dict",temp_dict)


          s8 = dt.datetime.now().replace(microsecond=0)
          print(f'format_topics_sentences and time taken is {s8-s7}')
          Tweets_doc=pd.DataFrame({'tweet_date':Tweets_documents['tweet_date'],'tweet_text':Tweets_documents["tweets"],'Topics':Tweets_documents['Topic_Keywords']})
          ideal_topic=5
          OverallTopics_tweets,Word_Freq=Word_Cloud_Freq(optimal_model,ideal_topic)
          s9 = dt.datetime.now().replace(microsecond=0)
          print(f'Word_Cloud_Freq and time taken is {s9-s8}')

          dictionary_tweets[days+1]={'Topic_Tweets_ID' : temp_dict,
                      'Tweets_for':
                          {
                          'company name':company,
                          'Best coherence score':coherence_lda,
                          'Overall topics': OverallTopics_tweets 
                      
                          },
                      'Topics and Word_Cloud':
                          {
                          'Tweet_documents':Tweets_doc.to_json(orient="records"),
                          'Word_cloud_freq':dict(Word_Freq)
                          }
                      }
          
          print(f"Tweets/{company}/{days} has been stored")
          
    
        except Exception as e:
            dictionary_tweets[days+1] = f'No news of {company} for {days} days'
            print(e)
            print(f'No tweets for {company} for {days} days')

        final_time = dt.datetime.now().replace(microsecond=0)
        print(f"************************************ Topic for day {days+1} and time take is {final_time - start_time} completed**************************************************")
    BUCKET.Object(BUCKET_NAME, f'Tweets/{company}.json').put(Body=json.dumps(dictionary_tweets,indent=1))
    return "Topics processed for Tweets"
    

def Topic_Modeling_News(df, company):
    # df = input_news_df
    df = df.dropna()
    df['New_date'] = pd.to_datetime(df.news_date, format='%Y-%m-%d')
    df['New_date'] = df['New_date'].dt.strftime('%Y-%m-%d')
    dictionary_news={}
  
    for days in range(DAYS):
      start_time = dt.datetime.now().replace(microsecond=0)
      print(f"************************************ For day {days+1}**************************************************")
      try:   
        date = (dt.datetime.strptime(TODAY, '%Y-%m-%d') - dt.timedelta(days)).strftime('%Y-%m-%d')
        df_News= df[((df['New_date'] >= date ))&(df['news_for']==company)]
        df_News=df_News.reset_index(drop=True)
        df_News['News']=df_News['long_description']
        text_list=df_News['long_description'] #.tolist()
        text_list = text_preprocessing_fn(text_list)
        tokenized_reviews=[x.split(" ") for x in text_list]
        s1 = dt.datetime.now().replace(microsecond=0)
        print(f'News cleaned and time taken (text preprocessing/cleaning) {s1-start_time}')
        tokenized_reviews=[x.split(" ") for x in text_list]
        s4 = dt.datetime.now().replace(microsecond=0)
        optimal_model,coherence_lda,corpus_tfidf=Optimal_Model(tokenized_reviews)
        s7 = dt.datetime.now().replace(microsecond=0)
        print(f'Optimal_Model and time taken is {s7-s4}')
        News_documents=format_topics_sentences(optimal_model,corpus_tfidf,df_News['News'],df_News['news_date'], df_News['news_ID'])


        # print("News doc",type(News_documents))
        # display(News_documents)
        # data_struct = lambda:defaultdict(list)
        temp_dict = defaultdict(list)
        for index,rows in News_documents.iterrows():
          temp_dict[rows['Topic_Keywords']].append(rows['news_ID'])

        print("temp_dict",temp_dict)



        s8 = dt.datetime.now().replace(microsecond=0)
        print(f'format_topics_sentences and time taken is {s8-s7}')
        News_doc=pd.DataFrame({'News_date':News_documents['news_date'],'News_text':News_documents["News"],'Topics':News_documents['Topic_Keywords']})
        ideal_topic=5
        OverallTopics_news,Word_Freq=Word_Cloud_Freq(optimal_model,ideal_topic)
        s9 = dt.datetime.now().replace(microsecond=0)
        print(f'Word_Cloud_Freq and time taken is {s9-s8}')

        dictionary_news[days+1]={'Topic_News_ID': temp_dict,  
                    'News_for':
                    {
                      'company name':company,
                      'Best coherence score':coherence_lda,
                      'Overall topics': OverallTopics_news
                    },
                    'Topics and Word_Cloud':
                    {
                        'News_documents':News_doc.to_json(orient="records"),
                        'Word_cloud_freq':dict(Word_Freq)
                    }
                  }
        
        print(f"News/{company}/{days} has been stored")
      except Exception as e:
        dictionary_news[days+1] = f'No news of {company} for {days} days'
        print(f'{e} No news of {company} for {days} days')

      final_time = dt.datetime.now().replace(microsecond=0)
      print(f"************************************ Topic for day {days+1} and time take is {final_time - start_time} completed**************************************************")
    print("Topics \n",dictionary_news)
  
    BUCKET.Object(BUCKET_NAME, f'News/{company}.json').put(Body=json.dumps(dictionary_news,indent=1))

    return "Topics processed for News"


def get_ticker_names_and_id(sector , table):
    ticker_id=[]
    ticker_name=[]
    response= table.scan(FilterExpression= Attr('Sector').eq(sector))
    for item in response['Items']:
        ticker_id.append(item['Stock_Ticker_Symbol'])
        ticker_name.append(item['Ticker_Name'])
    return ticker_id , ticker_name

  
def read_tweets_data(dynamodb_client, list_months):
  tweets_table = dynamodb_client.Table("Tweets_Table")
  items = []
  str_dict = {}
  str_dict[":val"]=True
  str_dict[":date"]=(datetime.now(timezone.utc) - dt.timedelta(30)+ dt.timedelta(hours=5 ,minutes=30)).strftime('%Y-%m-%d 00:00:00')
  for month in list_months:
      resp_ = tweets_table.query(Limit=100,
          KeyConditionExpression=Key("tweets_month").eq(month),
          FilterExpression="tweet_ner_flag= :val AND tweet_date > :date",
          ExpressionAttributeValues= str_dict
      )
      items.extend(resp_["Items"])

      while "LastEvaluatedKey" in resp_:
          try:
              resp_ = tweets_table.query(Limit=100,
                  ExclusiveStartKey=resp_["LastEvaluatedKey"],
                  KeyConditionExpression=Key("tweets_month").eq(month),
                  FilterExpression="tweet_ner_flag= :val AND tweet_date > :date",
                  ExpressionAttributeValues= str_dict
              )
              items.extend(resp_["Items"])
          except:
              time.sleep(5)
              resp_ = tweets_table.query(Limit=100,
                  ExclusiveStartKey=resp_["LastEvaluatedKey"],
                  KeyConditionExpression=Key("tweets_month").eq(month),
                  FilterExpression="tweet_ner_flag= :val AND tweet_date > :date",
                  ExpressionAttributeValues= str_dict
              )
              items.extend(resp_["Items"])
              
  tweets_dict = defaultdict(list)
  for item in items:
      tweets_dict[item["tweet_for"]].append(item)
  return dict(tweets_dict)

def read_news_data(dynamodb_client, list_months):
    news_table = dynamodb_client.Table("News_Table")
    items = []
    str_dict = {}
    str_dict[":val"]=True
    str_dict[":date"]=(datetime.now(timezone.utc) - dt.timedelta(30)+ dt.timedelta(hours=5 ,minutes=30)).strftime('%Y-%m-%d 00:00:00')
    for month in list_months:
        resp_ = news_table.query(Limit=50,
            KeyConditionExpression=Key("news_month").eq(month),
            FilterExpression="news_ner_flag= :val AND news_date > :date",
            ExpressionAttributeValues= str_dict
        )
        items.extend(resp_["Items"])

        while "LastEvaluatedKey" in resp_:
            try:
                resp_ = news_table.query(Limit=50,
                    ExclusiveStartKey=resp_["LastEvaluatedKey"],
                    KeyConditionExpression=Key("news_month").eq(month),
                    FilterExpression="news_ner_flag= :val AND news_date > :date",
                    ExpressionAttributeValues= str_dict,
                )
                items.extend(resp_["Items"])
            except:
                time.sleep(5)
                resp_ = news_table.query(Limit=50,
                    ExclusiveStartKey=resp_["LastEvaluatedKey"],
                    KeyConditionExpression=Key("news_month").eq(month),
                    FilterExpression="news_ner_flag= :val AND news_date > :date",
                    ExpressionAttributeValues= str_dict
                )
                items.extend(resp_["Items"])
            
    
    news_dict = defaultdict(list)
    for item in items:
        news_dict[item["news_for"]].append(item)
    return dict(news_dict)
    
if __name__ == "__main__":

    print("Entered in Topic Modelling Application")
    t1 = dt.datetime.now().replace(microsecond=0)
    table=algoDDB.Table('ConfigTable')
    this_month = (dt.datetime.now(timezone.utc)+dt.timedelta(hours=5, minutes = 30)).strftime("%Y-%m")
    today = dt.date.today()
    first = today.replace(day=1)
    lastMonth = first - dt.timedelta(days=1)
    last_month=lastMonth.strftime("%Y-%m")
    list_months=[this_month , last_month]
    IT_ticker_id , IT_ticker_name = get_ticker_names_and_id('IT', table)
    BANK_ticker_id , BANK_ticker_name = get_ticker_names_and_id('BANK', table)
    ticker_list = IT_ticker_id+BANK_ticker_id
    ticker_name = IT_ticker_name+BANK_ticker_name
    t2 = dt.datetime.now().replace(microsecond=0)
    print("Time to get Ticker :",t2-t1)
    print("Into read News")
    News = read_news_data(algoDDB,list_months)
    t3 = dt.datetime.now().replace(microsecond=0)
    print("Time to read news :",t3-t2)
    print("In Topic Prediction")
    t4 = dt.datetime.now().replace(microsecond=0)
    for i in ticker_name:
        s_tik = dt.datetime.now().replace(microsecond=0)
        News_data_list = []
        news = []
        print("#########################")
        print(f"************* Topic prediction on News for {i} *************")
        Topic_Modeling_News(pd.DataFrame(News[i]), i)
        e_tik = dt.datetime.now().replace(microsecond=0)
        print(f"************* Topic prediction on News for {i} completed in Time: {e_tik-s_tik} *************")
    t5 = dt.datetime.now().replace(microsecond=0)
    print("Time to get News Topic for all Ticker :",t5-t4)

    t6 = dt.datetime.now().replace(microsecond=0)
    print('Into read Tweets')
    Tweets = read_tweets_data(algoDDB,list_months)
    t7 = dt.datetime.now().replace(microsecond=0)
    print("Time to read tweets :",t7-t6)
    for j in ticker_name:
      s_tik = dt.datetime.now().replace(microsecond=0)
      Tweets_data_list = []
      tweets = []
      print("#########################")
      print(f"*************Topic prediction on Tweets for {j} *************")
      Topic_Modeling_Tweets(pd.DataFrame(Tweets[j]), j)
      e_tik = dt.datetime.now().replace(microsecond=0)
      print(f"************* Topic prediction on Tweets for {i} completed in Time: {e_tik-s_tik} *************")
    t8 = dt.datetime.now().replace(microsecond=0)
    print("Time to get Tweets Topic for all Ticker :",t8-t7)