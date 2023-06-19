from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import regex as re
import snscrape.modules.twitter as sntwitter
import datetime
from datetime import timezone, date
import asyncio
from asyncio import events



# Reading the contents of the relevent_word.txt file.
with open('Relevant_Words.txt') as rw:
    RELEVANT_WORDS = rw.read().lower().splitlines()


# Reading the contents of the irrelevent_word.txt file.
with open('Irrelevant_Words.txt') as iw:
    FILTER_OUT_WORDS = iw.read().lower().splitlines()


def contains_minimum_tokens(tweet_full_text, min_num_tokens, min_num_chars):
    """

    Function to check if tweet contains minimum(min_num_tokens) number of tokens or not.

    Parameters: 
    ----------
        tweet_full_text: str
            scraped tweet text
        min_num_tokens: int
            Minimum number of tokens a tweet should have. Defined in the config table.
        min_num_chars: int
            Minimum number of characters that a token should have. Defined in the config table.

    Returns:
    --------
        Bool: True if the tweet have minimum number of tokens, otherwise False.

    """
    
    #remove all the special characters
    text_for_tokens = re.sub(r"[^a-zA-Z0-9\s]", "", tweet_full_text) 

    # create tokens of words in text using nltk
    tokens = word_tokenize(text_for_tokens) 

    # convert the tokens to lower case
    tokens = [w.lower() for w in tokens] 
    
    # remove remaining tokens that are not alphabetic
    tokens = [w for w in tokens if w.isalpha()] 

    # filter out stop words
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if not w in stop_words]

    # remove words less than min_num_chars characters
    tokens = [w for w in tokens if not len(w) < min_num_chars]
    # print("NLTK Tokens: " + str(tokens))

    # returns False if the len of the list of tokens is less than the defined min_num_tokens given in the config table
    if len(tokens) < min_num_tokens:
        # print("Tweet does not contain min. number of tokens, not adding")
        return False
    
    return True

def filter_tweet(tweet_text):
    """

    Function to check if a given sentence contains at least one word or bigram or phrase
    from the RELEVANT_WORDS list and does not contain any word from the FILTER_OUT_WORDS list.
    
    Parameters: 
    ----------
        tweet_text: str
            scraped tweet text
    
    Returns:
    -------
        Bool: True if the sentence is relevant, False otherwise.

    """

    # splits the sentence into words and forms a list
    words = tweet_text.lower().split()

    # checks if any word is in the FILTER_OUT_WORDS list
    if any(word in FILTER_OUT_WORDS for word in words):
        return False
    
    # checks if any word is in the RELEVANT_WORDS list    
    if any(word.lower() in RELEVANT_WORDS for word in words):
        return True
    
    # checks if any bigram is in the RELEVANT_WORDS list
    for i in range(len(words)-1):
        bigram = words[i] + ' ' + words[i+1]
        if bigram.lower() in RELEVANT_WORDS:
            return True
        
    # checks if any phrases that 3 to 6 words long are in the RELEVANT_WORDS list    
    for i in range(len(words)):
        phrase3 = ' '.join(words[i:i+3])
        phrase4 = ' '.join(words[i:i+4])
        phrase5 = ' '.join(words[i:i+5])
        phrase6 = ' '.join(words[i:i+6])
        if (phrase3.lower() or phrase4.lower() or phrase5.lower() or phrase6.lower()) in RELEVANT_WORDS:
            return True
        
    return False

def contains_irrelavant_words(tweet_text, blacklist):
    """
    
    Function to check if tweet contains words from blacklist.

    Parameters: 
    ----------
        tweet_text: str
            scraped tweet text
        blacklist: str
            string present in the Exclusion_Tweet_String attribute of the config table
    
    Returns:
    -------
        Bool: True if the sentence contains word/words from the blacklist, False otherwise. 

    """

    if blacklist is None:
        return False
    
    blacklist = blacklist.split(", ")

    for pattern in blacklist:
        if re.search(pattern, tweet_text, re.IGNORECASE) != None:
            return True
        
    return False

async def update_tweet_table(Temp_Tweets_Table,tweet, ticker_name, datetime_now):
    '''

    Function to upsert the crawled tweets in the tweets_table

    Parameters
    ----------
        tweets_table : AWS DynamoDB Table
            "Tweets" Table from DynamoDB
        tweet : object 
            The tweets scraped (response of sntwitter.TwitterSearchScraper)
        ticker_name : Str 
            The string which will be used to store tweets along with the Ticker name in DynamoDB Table.
        datetime_now : date
            The current date and time at the time of scraping.

    '''

    tweets_month = (tweet.date+datetime.timedelta(hours=5,minutes=30)).strftime("%Y-%m")
    tweet_id = ticker_name + " +UNIQUE_T_AND_N_ID_SEPARATOR+ " + str(tweet)
    data = {
                # "id": str(uuid.uuid1()),
                "tweets_month": tweets_month,
                "tweets_ID": tweet_id,
                "tweet_for": ticker_name,
                "tweet_by": tweet.user.username,
                "tweet_date": (tweet.date+datetime.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                "crawled_date": (datetime_now+datetime.timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                "tweet_text": tweet.content,
                "tweet_link": str(tweet),
                "tweet_ner_flag": False,
                "events_detection_flag": False,
            }
    
    Temp_Tweets_Table.put_item(Item = data)   


def crawl_tweets_by_keyword(Temp_Tweets_Table, ticker_tweet_keywords, ticker_tweet_exclusions, 
                            ticker_name, tweet_followersCount, tweet_minimum_tokens, tweet_minimum_chars):
    '''

    Main Function for crawling tweets using snscrape library

    Parameters
    ----------
        tweets_table : AWS DynamoDB Table
            "Tweets" Table from DynamoDB
        ticker_tweet_keywords : Str 
            The query string which has to be passed to Sncscrape which will 
            return tweets about the specific company based on the keywords.
        ticker_tweet_exclusions : Str 
            The query string which has to be passed to Sncscrape which will not 
            return tweets about the specific company based on the exclusion string.
        ticker_name : Str 
            The string which will be used to store tweets along with the Ticker name in DynamoDB Table.
        tweet_followersCount : int
            The number used to store tweets only if the follower count of editor is 
            greater than the defined number.
        tweet_minimum_tokens : int
            The number used to store tweets only if the number of tokens of the tweet 
            is greater than the defined number.
        tweet_minimum_chars : int
            The number used to store tweets only if the number of characters of the tweet 
            is greater than the defined number.

    '''

    count = 0

    # Iterate through all of the Tweets returned by Snscrape
    for tweet in sntwitter.TwitterSearchScraper(
        f"({ticker_tweet_keywords}) lang:en since:"
        # + str(date(2023,4,10))
        + str(datetime.datetime.now(datetime.timezone.utc).date())
        + " -filter:retweets"
    ).get_items():
        
        #getting the current date and time
        datetime_now = datetime.datetime.now(timezone.utc)
        
        # Check if the followerCount of the editor is greater than the required amount
        if tweet.user.followersCount < tweet_followersCount:
            continue

        # Check if the tweet contains desired number of tokens
        if not contains_minimum_tokens(tweet.content, tweet_minimum_tokens, tweet_minimum_chars):
            continue

        # Check if the tweet contains irrelevant words
        if contains_irrelavant_words(tweet.content, ticker_tweet_exclusions):
            continue

        # Check if the tweet contains relevant words, if true update the tweet table   
        if filter_tweet(tweet.content) == True:
            count = count+1
            loop = asyncio.get_event_loop()
            loop.run_until_complete(update_tweet_table(Temp_Tweets_Table, tweet, ticker_name, datetime_now))            
            
    print(f"Number of tweets for {ticker_name} entered in the table : ",count)