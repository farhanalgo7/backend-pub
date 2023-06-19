import os
import json
import boto3
import datetime as dt
import pandas as pd
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import string
import spacy
import en_core_web_md
import gensim
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,strip_numeric
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel, CoherenceModel
from collections import Counter
# from fastapi import FastAPI
# from pydantic import BaseModel # for pydantic model
nltk.download('punkt')
nltk.download('stopwords')
#create model using VADER lexicon
nltk.download('vader_lexicon')
warnings.filterwarnings("ignore", category=DeprecationWarning)


#Code for sentiments
def sentiment(x):
    if x>= 0.05:
        return "Positive"
    elif x<= -0.05:
        return "Negative"
    else:
        return "Neutral"


##Part: pre-process the data
def clean_text(text): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))]) 
    
    return text2.lower()

#function to remove stopwords
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text

def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']): 
    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])
    output = []
    for sent in texts:
            doc = nlp(sent) 
            output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
    return output
 
# def lemmatization(texts): 
#   print("Enter lemeatization")
#   lemmatizer = nltk.stem.WordNetLemmatizer()
#   w_tokenizer = TweetTokenizer() 
#   output = []
  
#   print("Processing lemeatization")
#   for sent in texts:
#       output.append([(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((sent))])
  
#   print("Finished lemeatization")
#   return output

"""
##Part: Topic model
def LDAModel(tokenized_reviews):

    #create vocabulary and document term matrix
    dictionary = corpora.Dictionary(tokenized_reviews)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]
    #considering 2-40 topics 
    # num_topics=list(range(2,40+1,2)[1:])
    num_topics=list(range(2,10+1,2)[1:])


    num_keywords=10
    LDA_models={}
    LDA_topics={}
    for i in num_topics:
        LDA_models[i]=LdaModel(corpus=doc_term_matrix,id2word=dictionary,num_topics=i,update_every=1,chunksize=len(doc_term_matrix),passes=20,alpha='auto',random_state=123)
        shown_topics=LDA_models[i].show_topics(num_topics=i,num_words=num_keywords,formatted=False)
        LDA_topics[i]=[[word[0] for word in topic[1]] for topic in shown_topics]
    coherences=[CoherenceModel(model=LDA_models[i], texts=tokenized_reviews, dictionary=dictionary,coherence='c_v').get_coherence() for i in num_topics[:-1]]
    return dictionary,doc_term_matrix,num_topics,num_keywords,LDA_topics,LDA_models,coherences

#create function to derive jaccard similarity of two topics
def jaccard_similarity(topic_1,topic_2):
    intersection=set(topic_1).intersection(set(topic_2))
    union=set(topic_1).union(set(topic_2))
    return float(len(intersection))/float(len(union))

def Mean_Stability(num_topics,LDA_topics):

    LDA_stability={}
    for i in range(0,len(num_topics)-1):
        jaccard_sims=[]
        for t1, topic_1 in enumerate(LDA_topics[num_topics[i]]):
            sims=[]
            for t2, topic_2 in enumerate(LDA_topics[num_topics[i+1]]):
                sims.append(jaccard_similarity(topic_1, topic_2))
        jaccard_sims.append(sims)
        LDA_stability[num_topics[i]]=jaccard_sims
    mean_stabilities=[np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]
    
    return mean_stabilities

def Ideal_topic_num(coherences,mean_stabilities,num_keywords,num_topics):
    # coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(num_keywords)[:-1]] # limit topic numbers to the number of keywords
    print(coherences)
    coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(0,3,1)]
    coh_sta_max = max(coh_sta_diffs)
    coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
    ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
    ideal_topic_num = num_topics[ideal_topic_num_index]
    print("ideal_topic_num",ideal_topic_num)
    
    return ideal_topic_num
"""
def Optimal_Model(num_keywords,ideal_topic_num,tokenized_reviews):
  
    #Optimal Model 
    print("tokenized_reviews type", type(tokenized_reviews))
    dictionary = corpora.Dictionary(tokenized_reviews)
    print("In optimal model dictionary", type(dictionary) )
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]
    print("doc_term_matrix", type(doc_term_matrix))
    Optimal_model=LdaModel(corpus=doc_term_matrix,id2word=dictionary,num_topics=ideal_topic_num,update_every=1,chunksize=len(doc_term_matrix),passes=20,alpha='auto',random_state=123)
    print("LDA Model", Optimal_model, "/n", Optimal_model )
    model_topics = Optimal_model.show_topics(formatted=False)
    print("model_topics")
    coherence_model_lda = CoherenceModel(model=Optimal_model, texts=tokenized_reviews, dictionary=dictionary , coherence='c_v')
    print(" coherence_model_lda")
    coherence_lda = coherence_model_lda.get_coherence()
    print(" coherence_lda  Optimal_Model executed")
    return Optimal_model,coherence_lda,doc_term_matrix

    
def format_topics_sentences(model,corpus,Questions,Answers,Sentiments,Compound_score):
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
    Q1=pd.Series(Questions)
    A1=pd.Series(Answers)
    c1=pd.Series(Compound_score)
    s1=pd.Series(Sentiments)
    
    

    sent_topics_df = pd.concat([Q1,A1,c1,s1,sent_topics_df], axis=1)
    return(sent_topics_df)
#Sentiment_Analysis() calls internal function for sentimental analysis and analyse the sentiments on "Answers"
def Sentiment_Analysis(A):
    analyser = SentimentIntensityAnalyzer()
    scores_VADER = A.apply(lambda review: analyser.polarity_scores(review))
    compound_VADER = scores_VADER.apply(lambda score_dict: score_dict['compound'])
    Sentiments = compound_VADER.apply(sentiment)
    return Sentiments,compound_VADER

#Converting all the outputs into dataframe to give input to Topic modeling() 
def build_dataframe(DF):
    Compound_score=Sentiment_Analysis(DF['Answers'])[1]
    Sentiments=Sentiment_Analysis(DF['Answers'])[0]
    col_names=['Questions','Answers','Compound_score','Sentiments']
    df=pd.DataFrame(columns=col_names)
    df['Questions']=DF['Questions']
    df['Answers']=DF['Answers']
    df['Compound_score']=Compound_score
    df['Sentiments']=Sentiments
    Total_Pos=len(df[df['Sentiments']=='Positive'])
    Total_Neg=len(df[df['Sentiments']=='Negative'])
    Total_Neu=len(df[df['Sentiments']=='Neutral'])
    return df,Total_Pos,Total_Neg,Total_Neu

#Topic Modeling for Positive documents
def Topic_Modeling_Positive(df):
    df_Pos=df[df['Sentiments']=='Positive']
    df_Pos=df_Pos.reset_index(drop=True)
    df_Pos['Original_text']=df['Answers']
    df_Pos['Answers'] = df_Pos['Answers'].apply(clean_text)
    # remove stopwords from the text
    df_Pos['Answers']=df_Pos['Answers'].apply(remove_stopwords)
    text_list=df_Pos['Answers'].tolist()
    ss = " ".join(text_list)
    print("word count", len(ss.split()))
    tokenized_reviews = lemmatization(text_list)
    print("tokenized_reviews completed")
    #dictionary,doc_term_matrix,num_topics,num_keywords,LDA_topics,LDA_models,coherences=LDAModel(tokenized_reviews)
    #mean_stabilities=Mean_Stability(num_topics,LDA_topics)
    #ideal_topic=Ideal_topic_num(coherences,mean_stabilities,num_keywords,num_topics)
    ideal_topic=4
    num_keywords=10
    print("at Optimal_Model ")
    optimal_model,coherence_lda,doc_term_matrix=Optimal_Model(num_keywords,ideal_topic,tokenized_reviews)
    print("Process at Optimal_Model  completed")
    OverallTopics_Pos=optimal_model.print_topics(num_words=num_keywords)
    Positive_documents=format_topics_sentences(optimal_model,doc_term_matrix,df_Pos['Questions'],df_Pos['Original_text'],df_Pos['Sentiments'],df_Pos['Compound_score'])
    return optimal_model,Positive_documents,OverallTopics_Pos,ideal_topic,coherence_lda

#Topic modeling for Negative documents
def Topic_Modeling_Negative(df):
    df_Neg=df[df['Sentiments']=='Negative']
    df_Neg=df_Neg.reset_index(drop=True)
    df_Neg['Original_text']=df['Answers']
    df_Neg['Answers'] = df_Neg['Answers'].apply(clean_text)
    df_Neg['Answers']=df_Neg['Answers'].apply(remove_stopwords) # remove stopwords from the text
    text_list=df_Neg['Answers'].tolist()
    tokenized_reviews = lemmatization(text_list)
    print("tokenized_reviews completed")
    ss = " ".join(text_list)
    print("word count", len(ss.split()))
    #dictionary,doc_term_matrix,num_topics,num_keywords,LDA_topics,LDA_models,coherences=LDAModel(tokenized_reviews)
    #mean_stabilities=Mean_Stability(num_topics,LDA_topics)
    #ideal_topic=Ideal_topic_num(coherences,mean_stabilities,num_keywords,num_topics)
    ideal_topic=2
    num_keywords=10
    print("at Optimal_Model ")
    optimal_model,coherence_lda,doc_term_matrix=Optimal_Model(num_keywords,ideal_topic,tokenized_reviews)
    print("Process at Optimal_Model  completed")
    OverallTopics_Neg=optimal_model.print_topics(num_words=num_keywords)
    Negative_documents=format_topics_sentences(optimal_model,doc_term_matrix,df_Neg['Questions'],df_Neg['Original_text'],df_Neg['Sentiments'],df_Neg['Compound_score'])
    return optimal_model,Negative_documents,OverallTopics_Neg,ideal_topic,coherence_lda

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

#get_result() gives output in the dictionary format
def get_result(input_csv_qa,Filename):
    s1 = dt.datetime.now().replace(microsecond=0)  
    DF = pd.read_csv(input_csv_qa,sep=",",encoding = 'cp1252')
    print(DF.columns)
    s2 = dt.datetime.now().replace(microsecond=0)
    df,overall_Pos,overall_Neg,overall_Neu=build_dataframe(DF)
    s3 = dt.datetime.now().replace(microsecond=0)  
    print(f'Time taken to build dataframe = {s3-s2}')
    model_Pos,Positive_documents,Overall_Topics_Pos,ideal_topic_Pos,coherence_lda_Pos=Topic_Modeling_Positive(df)
    s4 = dt.datetime.now().replace(microsecond=0)  
    print(f'Time taken for Topic_Modeling_Positive = {s4-s3}')
    total_pos,Freq_Pos=Word_Cloud_Freq(model_Pos,ideal_topic_Pos)
    s5 = dt.datetime.now().replace(microsecond=0)  
    print(f'Time taken for Word_Cloud_Freq (Positive) = {s5-s4}')
    Pos=pd.DataFrame({'Questions':Positive_documents["Questions"],'Answers':Positive_documents["Original_text"],'Compound_score':Positive_documents["Compound_score"],'Sentiments':Positive_documents["Sentiments"],'Topics':Positive_documents["Topic_Keywords"]})
    s6 = dt.datetime.now().replace(microsecond=0)
    print(f'Time taken to build Pos dataframe = {s6-s5}')
    model_Neg,Negative_documents,Overall_Topics_Neg,ideal_topic_Neg,coherence_lda_Neg=Topic_Modeling_Negative(df)
    s7 = dt.datetime.now().replace(microsecond=0)
    print(f'Time taken for Topic_Modeling_Negative = {s7-s6}')
    total_neg,Freq_Neg=Word_Cloud_Freq(model_Neg,ideal_topic_Neg)
    s8 = dt.datetime.now().replace(microsecond=0)
    print(f'Time taken for Word_Cloud_Freq (Negative) = {s8-s7}')

    Neg=pd.DataFrame({'Questions':Negative_documents["Questions"],'Answers':Negative_documents["Original_text"],'Compound_score':Negative_documents["Compound_score"],'Sentiments':Negative_documents["Sentiments"],'Topics':Negative_documents["Topic_Keywords"]})
    s9 = dt.datetime.now().replace(microsecond=0)
    print(f'Time taken to build Neg dataframe = {s9-s8}')
    df3=df[df['Sentiments']=='Neutral']
    Overall_documents=pd.concat([Pos, Neg,df3], ignore_index=True, sort=False)
    Documents=Overall_documents.to_json(orient='records')
    s10 = dt.datetime.now().replace(microsecond=0)
    print(f'Total Time = {s10-s1}')
    return {Filename:
            {
                'Overall_Sentiments':
            {
                'Positive':overall_Pos,
                'Negative':overall_Neg,
                'Neutral':overall_Neu
                },
            'Overall_topics':
            {
                'Positive':total_pos,
                    'Best topic numbers(Positive)':ideal_topic_Pos,
                    'coherence score for Positive documents':coherence_lda_Pos,
                    'Negative':total_neg,
                    'Best topic numbers (Negative)':ideal_topic_Neg,
                    'coherence score for Negative documents':coherence_lda_Neg
                }
            },
            'Output':
            {
            'Overall_Documents':Documents,
            'WordCloud':
            {
                'Positive_WordCloud':dict(Freq_Pos),
                'Negative_WordCloud':dict(Freq_Neg)  
            }
            }
    }



if __name__=="__main__":
  #AWS Credentials
  REGION = "ap-south-1"
  BUCKET_NAME = "temp-backup-tweets-and-news"
  ACCESS_KEY_ID = 'AKIA4YOVCNBKS2LXNQFW'
  SECRET_ACCESS_KEY = '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'
  BUCKET = boto3.resource('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)


  start = dt.datetime.now().replace(microsecond=0)
  directory = "data/"
  for f in os.listdir(directory):
    csv = os.path.join(directory, f)
    if os.path.isfile(csv):
      t1 = dt.datetime.now().replace(microsecond=0)
      print(f"<<<<<<<< Processing for {f} >>>>>>>>>>>>>")
      txt = f.replace(".csv",'')
      txt = txt.split('-')
      print(txt)
      try:
        dictionary = get_result(csv,'file')
        print(f"results for {f} is {dictionary} \n")
        BUCKET.Object(BUCKET_NAME, f'Earning-Calls/{txt[0]}/{txt[1]}-{txt[2]}.json').put(Body=json.dumps(dictionary,indent=1))
        BUCKET.Object(BUCKET_NAME, f'Earning-Calls/{txt[0]}/{txt[1]}-{txt[2]}.csv').put(Body=open(csv,'rb'))
    
        print("##########################")
      except Exception as e:
        print(f"results for {f} error occured {e}")
        print("##########################")

      t2 = dt.datetime.now().replace(microsecond=0)
      print("Time to process for single transcripts",t2-t1)

  end = dt.datetime.now().replace(microsecond=0)
  print("Total time to process for all transcripts",end-start)