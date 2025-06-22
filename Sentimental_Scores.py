import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Twitter API credentials (replace with your own)
api_key = 'Sy9ngytZftYKvKVILHiJYaKCf'
api_secret_key = 'k4k8yT3mcAoz5i5EIkx9ZZLlYdOLowfrrDVENjQQwgyYPodyAe'
access_token = '1823564825248804864-hLVkIMqtiKEc4EXIjrjD8Afj3M4KVI'
access_token_secret = 'TSUc91Hc2jcDiPUE4DJ9Uj5grSr7FCRxUel9AkQZuqjtW'

# Authentication
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Gather tweets
tweets = tweepy.Cursor(api.search_tweets, q='AAPL', lang='en', since='2024-01-01').items(100)

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
data = []

for tweet in tweets:
    text = tweet.text
    sentiment = sid.polarity_scores(text)['compound']
    data.append({'Date': tweet.created_at, 'Text': text, 'Sentiment': sentiment})

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel('sentiment_data.xlsx', index=False)
