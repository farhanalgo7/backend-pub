import requests
import json
import pytest

BASE_URL = "http://localhost:8000/"
API_URL = "http://localhost:8000/search/"

headers = {'Content-type': 'application/json'}

data = dict()


def test_health_check():
    response = requests.get(url=BASE_URL)
    assert response.status_code == 200


def test_semantic_search_pipeline():
    search_string = "TCS"

    response = requests.get(url = API_URL + search_string, headers = headers)
    assert response.status_code == 200


def test_question_answer_pipeline():
    search_string = "What is the interest rate for SBI?"

    response = requests.get(url = API_URL + search_string, headers = headers)
    assert response.status_code == 200


def test_semantic_search_news_response():
    search_string = "TCS"

    response = requests.get(url = API_URL + search_string, headers = headers)
    assert response.status_code == 200

    api_response = json.loads(response.text)

    # Check each key for News response to make sure the correct info is passed to the UI
    assert "News_Search_Results" in api_response
    assert "search_result" in api_response.get("News_Search_Results")

    # Get results in the form of content0 : <News>
    search_results = api_response.get("News_Search_Results").get("search_result")

    for i, value in enumerate(search_results.items()):
        assert isinstance(value[1], str)
        assert value[1] is not "" and value[1] is not None
    

    # Get probability scores in the form of content0_score : 0.6801
    scores_results = api_response.get("News_Search_Results").get("score")

    for i, value in enumerate(scores_results.items()):
        assert isinstance(value[1], float)
        assert value[1] is not None and value[1] > 0.0
    

    # Get news items in the form of news : news{}
    news_results = api_response.get("News_Search_Results").get("news")

    # Additional validation for news is not added since this is out of our scope
    for news_ids, news_attributes in news_results.items():
        assert "News_Title" in news_attributes
        assert "News_Source" in news_attributes
        assert "News_Sentiment" in news_attributes
        assert "News_Time_Stamp" in news_attributes
        assert "News_Sentiment_Score" in news_attributes
        assert "News_Summary" in news_attributes
        assert "News_Tags" in news_attributes
        assert "News_ID" in news_attributes
        assert "ID" in news_attributes
        assert "Events_Detected" in news_attributes
        assert "News" in news_attributes
        assert "Actual_News" in news_attributes
        assert "News_date" in news_attributes
        assert "Sentiment_Analyzed_news" in news_attributes


def test_semantic_search_tweets_response():
    search_string = "TCS"

    response = requests.get(url = API_URL + search_string, headers = headers)
    assert response.status_code == 200

    api_response = json.loads(response.text)

    # Check each key for News response to make sure the correct info is passed to the UI
    assert "Tweets_Search_Results" in api_response
    assert "search_result" in api_response.get("Tweets_Search_Results")

    # Get results in the form of content0 : <News>
    search_results = api_response.get("Tweets_Search_Results").get("search_result")

    for i, value in enumerate(search_results.items()):
        assert isinstance(value[1], str)
        assert value[1] is not "" and value[1] is not None
    

    # Get probability scores in the form of content0_score : 0.6801
    scores_results = api_response.get("Tweets_Search_Results").get("score")

    for i, value in enumerate(scores_results.items()):
        assert isinstance(value[1], float)
        assert value[1] is not None and value[1] > 0.0
    

    # Get news items in the form of tweets : tweets{}
    tweets_results = api_response.get("Tweets_Search_Results").get("tweets")

    # Additional validation for news is not added since this is out of our scope
    for tweet_ids, tweet_attributes in tweets_results.items():
        assert "Tweets_Title" in tweet_attributes
        assert "Tweets_Source" in tweet_attributes
        assert "Tweets_Sentiment" in tweet_attributes
        assert "Tweets_Time_Stamp" in tweet_attributes
        assert "Tweets_Sentiment_Score" in tweet_attributes
        assert "Tweets_Summary" in tweet_attributes
        assert "Tweets_Tags" in tweet_attributes
        assert "Tweets_ID" in tweet_attributes
        assert "ID" in tweet_attributes
        assert "Events_Detected" in tweet_attributes
        assert "Tweets" in tweet_attributes
        assert "Actual_Tweets" in tweet_attributes
        assert "Tweets_date" in tweet_attributes
        assert "Sentiment_Analyzed_tweets" in tweet_attributes