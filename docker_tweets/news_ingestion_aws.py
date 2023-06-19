import datetime
from newspaper import Article
from pygooglenews import GoogleNews
import w3lib.html
import w3lib
from datetime import timezone
import requests
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
import asyncio
import pycld2 as cld2

# getting the current date, time and month at the time of crawling
datetime_now = datetime.datetime.now(timezone.utc)
news_month = (datetime_now+datetime.timedelta(hours=5,minutes=30)).strftime("%Y-%m")

def Find_url(string):
    """

    Function to find if a url is present in news article.
    This function will be used for cleaning news articles from Business Standard source, as Business standard
    returns unnecessary content while scrapping before the main article.

    Prameters: 
    ----------
        string : str
            We will pass the scrapped article from business standard and find the first link.

    Returns: 
    --------
        List: It returns a list of urls from the string

    """

    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


def get_content(link):
    """

    Fuction that returns long description (content) of news crawled
    Uses Article library from newspaper
    
    Parameters:
    -----------
        link : str
            The link or url of the article 

    Returns:
    --------
        text: str
            content of the article

    """

    try:
        # crawling and fetching the news content
        article = Article(link)
        article.download()
        article.parse()
    except:
        return False

    return article.text

def news_lang_check(description):
    """

    Function to check if the language of the news article 
    we want to add only those news which are of english language

    Parameters: 
    -----------
        long_description - str
            content of the article

    Returns:
    --------
        bool: True if the language of the article is English, else False.

    """

    printable_str = ''.join(x for x in description if x.isprintable())
    isReliable, textBytesFound, details, vector = cld2.detect(printable_str, returnVectors=True)

    languages = ['TELUGU', 'HINDI', 'KOREAN']

    if any(element[2] in languages for element in vector):
        return False
    else:
        return True

async def add_news_article(news_table, news_id, ticker_symbol, item, long_description):
    """

    Function to add news articles to news table
    we want to add only those news with news content present in them hence we will call 
    this function inside every if - elif condition

    Parameters: 
    -----------
        news_table - cosmosdb table object
        news_id - str:
            Unique id which is assigned to every article
        ticker_symbol - str:
            ticker symbol of the company
        item - response of pygooglenews api
        long_description - str
            content of the article

    """
    # print(item)

    try:
        long_desc = news_lang_check(long_description)

        if long_desc == True:
        # store all news related info in defined format
            news = {
                    # "id": str(uuid.uuid1()),
                    "news_month": news_month,
                    "news_ID": news_id,
                    "crawled_date": (datetime_now+datetime.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                    "crawled_source": "pygooglenews",
                    "news_for": ticker_symbol,  # company Name
                    "news_source": item.source["title"],  # Source of the article
                    "news_date": str((datetime.datetime.strptime(item.published, "%a, %d %b %Y %H:%M:%S GMT") 
                                    + datetime.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")), # Date of the article (datetime)
                    "link": item.link,  # Link of the article
                    "title": item.title,  # Title of the artcile
                    "long_description": long_description,  # Content of the article
                    'short_description': w3lib.html.remove_tags(item.summary),
                    "news_ner_flag": False,  # NER Flag for Data Enrichment Check
                    "events_detection_flag": False,
                }
            
            if news['news_for']=='Tata Consultancy Services Ltd':
                if 'Tata' in news['long_description']:
                    # print(news)
                    # print(f'news for {ticker_symbol}') 
                    news_table.put_item(Item = news)
                else:
                    pass
                
            elif "Container Store" in news['title']:
                pass
                    
            else:
                # print(news)
                # print(f'news for {ticker_symbol}')           
                news_table.put_item(Item = news)
        
        else:
            pass
            # print(item.source["title"])
            # print(long_description)
            
    except Exception as e:
        # print("Duplicated Value")
        print("Error in add_news_article function: ", e)


def crawl_news_by_keyword(news_table, frequency, ticker_news_keywords, ticker_symbol, sector):
    '''

    Main Function for crawling news using pygooglenews library

    Parameters
    ----------
        news_table : AWS DynamoDB Table
            "News" Table from DynamoDB
        frequency : Str 
            The time duration from which the user wants to get the news
        ticker_news_keywords : Str 
            The query string which has to be passed to google news which will return news about 
            the specific company.
        ticker_symbol : Str 
            The string which will be used to store news along with the Ticker name in DynamoDB Table.
        sector: Str
            The Sector which each ticker belongs to.

    '''

    # Call GoogleNews, country = India, lang = 'en'
    gn = GoogleNews(lang = 'en')  
    
    # Query to fetch the news, it will search in the title
    if sector == 'CONSTRUCTION':
        result = gn.search(f"{ticker_news_keywords}", when=frequency)
    else:    
        result = gn.search(f"allintitle:{ticker_news_keywords}", when=frequency)

    # storing the news entries recieved from pygooglenews
    newsitem = result["entries"]
    count = 0

    # iterate over the news entries
    for item in newsitem:
        news_id = ticker_symbol + " +UNIQUE_T_AND_N_ID_SEPARATOR+ " + item.link

        try:
            long_description = get_content(item.link)

            if long_description=="" or long_description==False:
                # print(item.link)

                if item.source['title'] == 'Indiainfoline':
                    news = requests.get(item.link)
                    soup = BeautifulSoup(news.content, 'html.parser')
                    article_text = soup.find(class_='widget-content cms-content ng-star-inserted')
                    long_description = article_text.text
                    # print(f"News for {ticker_symbol}")
                    # print("In Indiainfoline:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1

                elif item.source['title'] == 'CNBCTV18':

                    #creating a regex pattern to find classes which have the pattern in them.
                    news_class_pattern = re.compile(r'.*(art-page-wrapper|articleWrap).*')

                    news = requests.get(item.link)
                    soup = BeautifulSoup(news.content, 'html.parser')
                    article_text = soup.find("div", {"class" : news_class_pattern})

                    # remove tags that contain recommended news from the div extracted
                    if article_text:
                        for section in article_text.select("section"):
                                section.extract()  

                    long_description = article_text.get_text().strip()
                    # print(f"News for {ticker_symbol}")
                    # print("In CNBCTV18:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1

                elif item.source['title'] == 'Press Trust of India':
                    news = requests.get(item.link)
                    soup = BeautifulSoup(news.content, 'html.parser')
                    article_text = soup.find(class_ = "big-image-description")
                    long_description = article_text.get_text().strip()
                    # print(f"News for {ticker_symbol}")
                    # print("In Press Trust of India:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1
                
                elif item.source['title'] == 'The Tribune India':
                    req = Request(item.link, headers={'User-Agent': 'Mozilla/5.0'})
                    webpage = urlopen(req).read()
                    soup = BeautifulSoup(webpage, 'html.parser')
                    article_text = soup.find(class_ = 'story-desc')
                    long_description = article_text.get_text().strip()
                    # print(f"News for {ticker_symbol}")
                    # print("In The Tribune India:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1

                elif item.source['title'] == 'Business Standard':
                    news = requests.get(item.link)
                    soup = BeautifulSoup(news.content, 'html.parser')
                    article_text = soup.find(class_ = "storycontent")

                    # remove tags that contain recommended news from the div extracted
                    if article_text:
                        for tag in article_text.select("div.storycontent > div:nth-child(1) > div.mb-20"):
                            tag.extract()

                    long_description = article_text.get_text().strip()
                    # print(f"News for {ticker_symbol}")
                    # print("In Business Standard:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1

                elif item.source['title'] == 'Outlook India':
                    req = Request(item.link, headers={'User-Agent': 'Mozilla/5.0'})
                    webpage = urlopen(req).read()
                    soup = BeautifulSoup(webpage, 'html.parser')
                    article_text = soup.find(id = 'articleBody')

                    # remove tags that contain recommended news from the div extracted
                    if article_text:
                        for tag in article_text.select("#articleBody > div.__related_stories_thumbs"):
                            tag.extract()
                    
                    long_description = article_text.get_text().strip()
                    # print(f"News for {ticker_symbol}")
                    # print("In Outlook India:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1

                elif item.source['title'] == 'Investing.com India':
                    req = Request(item.link, headers={'User-Agent': 'Mozilla/5.0'})
                    webpage = urlopen(req).read()
                    soup = BeautifulSoup(webpage, 'html.parser')
                    article_text = soup.find(id = 'article-item-content')                   
                    long_description = article_text.get_text().strip()
                    # print(f"News for {ticker_symbol}")
                    # print("In Investing.com India:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1   

                elif item.source['title'] == 'Economic Times':
                    req = Request(item.link, headers={'User-Agent': 'Mozilla/5.0'})
                    webpage = urlopen(req).read()
                    soup = BeautifulSoup(webpage, 'html.parser')
                    article_text = soup.find(class_ = "artData clr")

                    # remove tags that contain recommended news from the div extracted
                    if article_text:
                        for tag in article_text.select("div.mb20.flt"):
                            tag.extract()      

                    long_description = article_text.get_text().strip()
                    # print(f"News for {ticker_symbol}")
                    # print("In Economic Times:")
                    # print(long_description)
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                    count = count+1       
                
            else:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(add_news_article(news_table, news_id, ticker_symbol, item, long_description))
                count = count+1
            
        
        #     elif item.source['title'] == 'Forbes India':
        #         req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
        #         webpage = urlopen(req).read()
        #         htmlParse = BeautifulSoup(webpage, 'html.parser')
        #         for para in htmlParse.find_all('p'):
        #             art = art + ' ' + para.text.strip()
        #         long_description = art
        #         add_news_article(news_table, news_id,
        #                          ticker_symbol, item, long_description)
        #         count = count+1
   

        except Exception as e:
            print("Error in crawl_news_by_keyword function: ", e)
          
    print(f"Number of news for {ticker_symbol} entered in the table : ", count)
