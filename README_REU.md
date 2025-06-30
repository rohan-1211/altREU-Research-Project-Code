# Stock Market Prediction Using Sentiment Analysis

This research project explores the relationship between public sentiment and stock price movement using a machine learning-based approach. It integrates sentiment scores from news headlines and financial text with historical stock data to predict price trends.

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
- `nltk`, `textblob`, `vaderSentiment`

## Additional Resources

- [Final Research Paper (PDF)](./altREU_Research_Paper_Final.pdf)
- [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment)
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
- [Yahoo Finance Python API](https://pypi.org/project/yfinance/)
