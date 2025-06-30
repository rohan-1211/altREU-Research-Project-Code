# Stock Market Prediction Using Sentiment Analysis

This project uses machine learning to predict stock prices by combining historical market data with sentiment analysis from news articles and social media. We implemented models that focused on Linear Regression as the main supervised machine learning algorithm, with performance evaluated using Mean Absolute Error (MAE). The integration of sentiment data collected using VADER and scraped using BeautifulSoup improved prediction accuracy by 3.2%, demonstrating the value of incorporating qualitative data alongside traditional financial metrics.

The project was developed during the altREU research program at Portland State University, and focuses on major technology stocks such as Apple, Microsoft, and Google. Full methodology, results, and analysis are detailed in the following research paper.

## Overview

The project collects historical stock prices and correlates them with sentiment data derived from textual analysis. It uses Natural Language Processing (NLP) to compute sentiment scores from news articles and feeds this data into predictive models. The final implementation includes multiple scripts for data processing, sentiment scoring, and modeling.

## Features

- Sentiment analysis using VADER and TextBlob
- Historical stock data collection via Yahoo Finance
- Sentiment score aggregation and normalization
- Machine learning model for stock movement prediction
- Modular Python scripts for analysis and reproducibility

## Project Structure

- `Stock_Data.py` – Fetches historical stock prices  
- `Sentiment_Analysis.py` – Extracts sentiment from financial headlines  
- `Sentimental_Scores.py` – Computes and organizes daily sentiment scores  
- `ML_Model.py` – Builds and evaluates machine learning models using stock + sentiment data  
- `ML_Model_SentimentScores.py` – Alternative version using preprocessed sentiment scores  
- `altREU_Research_Paper_Final.pdf` – Final research paper summarizing methodology and findings

## Tools and Libraries

- Python 3.x  
- `pandas`, `numpy`, `matplotlib`  
- `scikit-learn`  
- `yfinance`  
- `nltk`, `textblob`, `vaderSentiment`, `BeautifulSoup`

## Additional Resources

- [Final Research Paper (PDF)](./altREU_Research_Paper_Final.pdf)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [Yahoo Finance Python API](https://pypi.org/project/yfinance/)
