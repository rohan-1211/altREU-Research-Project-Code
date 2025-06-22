from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# URL for Finviz
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'INTC', 'CSCO']

# Dictionary to hold news tables
news_tables = {}

# Fetch news tables for each ticker
for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    
    # Parse the HTML content using BeautifulSoup
    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# Parse the fetched news data
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        if row.a:
            title = row.a.get_text()
            date_data = row.td.text.strip().split(' ')
            
            if len(date_data) == 1:
                time = date_data[0]
                if parsed_data:
                    date = parsed_data[-1][1]
                else:
                    date = ''
            else:
                date = date_data[0]
                time = date_data[1]
            
            # Handle 'Today' and 'Yesterday'
            if date == "Today":
                date = datetime.today().date()
            elif date == "Yesterday":
                date = datetime.today().date() - timedelta(days=1)
            elif isinstance(date, str):
                date = datetime.strptime(date, '%b-%d-%y').date()
                
            parsed_data.append([ticker, date, time, title])

# Convert parsed data to DataFrame
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Apply sentiment analysis
df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])

# Convert dates to datetime format
df['date'] = pd.to_datetime(df['date'])

# Calculate mean sentiment scores by ticker and date
mean_df = df.groupby(['ticker', df['date'].dt.date]).mean(numeric_only=True)
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()

# Filter the DataFrame to include only the last 6 days
last_6_days = mean_df.loc[mean_df.index >= (datetime.today().date() - timedelta(days=6))]

# Print the mean sentiment scores as a table
print("Mean Sentiment Scores by Ticker and Date (Last 6 Days):")
print(last_6_days.to_string(index=True, header=True))

# Plot the filtered data
last_6_days.plot(kind='bar', figsize=(12, 8))

# Improve date formatting on x-axis
plt.xticks(rotation=45, ha='right')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.title('Average Sentiment Score by Ticker and Date')
plt.tight_layout()

plt.show()
