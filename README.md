# Stock Price Prediction and Sentiment Analysis

This project combines stock price prediction using machine learning models (LSTM) and sentiment analysis based on news headlines. The application is built using Streamlit for an interactive user interface.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Sources](#data-sources)
6. [Model Architecture](#model-architecture)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Results](#results)
9. [Instructions](#instructions)

## Overview
This project predicts stock price movements using a Long Short-Term Memory (LSTM) neural network. Additionally, it incorporates sentiment analysis from news headlines to provide a comprehensive market outlook.

## Features
- Download historical stock price data for Nifty 50 using Yahoo Finance.
- Perform technical analysis by calculating indicators such as moving averages, RSI, Bollinger Bands, MACD, and ATR.
- Build and train an LSTM model for stock price prediction.
- Fetch and analyze news headlines for sentiment using the News API and TextBlob.
- Visualize correlation heatmaps and prediction results.

## Installation

### Prerequisites
Ensure you have Python 3.7 or higher installed.

### Required Libraries
Install the necessary libraries using the following command:
```bash
pip install yfinance numpy pandas matplotlib seaborn streamlit requests textblob keras scikit-learn
```

### Additional Setup
Install NLTK corpora required by TextBlob:
```bash
python -m textblob.download_corpora
```

## Usage

### Running the Application
1. Clone this repository or download the project files.
2. Open a terminal and navigate to the project directory.
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open the provided URL in your browser.

### User Interface
- View historical stock data with technical indicators.
- Visualize a correlation heatmap.
- Train the LSTM model and view actual vs predicted prices.
- Fetch and analyze news headlines for sentiment.

## Data Sources
- **Stock Price Data:** [Yahoo Finance](https://finance.yahoo.com)
- **News Headlines:** [NewsAPI](https://newsapi.org) (Replace with your API key)

## Model Architecture
The LSTM model used has the following architecture:
- LSTM layer with 128 units (return sequences: True)
- Dropout layer with 0.2 probability
- LSTM layer with 64 units (return sequences: False)
- Dropout layer with 0.2 probability
- Dense layer with 1 unit

## Sentiment Analysis
Sentiment analysis is performed using TextBlob, which assigns polarity scores to news headlines.

## Results
- Mean Squared Error (MSE) and Mean Absolute Error (MAE) are computed to evaluate the LSTM model.
- News sentiment is displayed as Positive, Negative, or Neutral.
- The final prediction combines price movement and sentiment.

## Instructions

### Setting API Key
Replace the placeholder API key in the following line with your NewsAPI key:
```python
api_key = 'your_api_key_here'
```

### Model Training
- The LSTM model is trained on 80% of the data.
- Model evaluation metrics such as MSE and MAE are displayed.

### News Sentiment Analysis
- Fetches the latest 10 news articles for the "Nifty 50" ticker.
- Performs sentiment analysis and displays overall sentiment.

### Visualizations
- Correlation heatmap of stock features.
- Actual vs predicted stock prices using LSTM.

### Final Prediction
- Combines LSTM prediction and sentiment analysis for market outlook.

## Notes
- Ensure a stable internet connection for data fetching from Yahoo Finance and NewsAPI.
- The application is designed for educational purposes and should not be used for actual trading decisions.

