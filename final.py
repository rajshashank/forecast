import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to fetch news using NewsAPI


def fetch_news(ticker, api_key, num_articles=10):
    url = f'https://newsapi.org/v2/everything?q="{ticker}"&apiKey={api_key}'
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"Error: {response.status_code}, {response.text}")
        return [], []

    news_data = response.json()
    articles = news_data.get('articles', [])[:num_articles]
    news_titles = [article['title'] for article in articles]

    return news_titles

# Function to perform sentiment analysis using TextBlob


def analyze_sentiment(titles):
    sentiment_scores = []
    for title in titles:
        score = TextBlob(title).sentiment.polarity
        sentiment_scores.append(score)
    return sentiment_scores


# Streamlit UI
st.title('Stock Price Prediction and Sentiment Analysis')

# Download Nifty 50 Data
n50 = yf.download("^NSEI", period="max", interval="1d")
n50.dropna(inplace=True)

# Feature Engineering
n50['Daily Return'] = n50['Close'].pct_change()
n50['10ma'] = n50['Close'].rolling(window=10).mean()
n50['50ma'] = n50['Close'].rolling(window=50).mean()
n50['100ma'] = n50['Close'].rolling(window=100).mean()
n50['200ma'] = n50['Close'].rolling(window=200).mean()
n50['Volatility'] = n50['Daily Return'].rolling(window=10).std()

# Calculate RSI
delta = n50['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
n50['RSI'] = 100 - (100 / (1 + rs))

# Calculate Bollinger Bands
n50['20-Day SMA'] = n50['Close'].rolling(window=20).mean()
n50['20-Day StdDev'] = n50['Close'].rolling(window=20).std()
n50['Upper Band'] = n50['20-Day SMA'] + (2 * n50['20-Day StdDev'])
n50['Lower Band'] = n50['20-Day SMA'] - (2 * n50['20-Day StdDev'])

# Calculate MACD
n50['12-Day EMA'] = n50['Close'].ewm(span=12, adjust=False).mean()
n50['26-Day EMA'] = n50['Close'].ewm(span=26, adjust=False).mean()
n50['MACD'] = n50['12-Day EMA'] - n50['26-Day EMA']
n50['Signal Line'] = n50['MACD'].ewm(span=9, adjust=False).mean()

# Calculate ATR
n50['TR'] = np.maximum((n50['High'] - n50['Low']),
                       np.maximum(abs(n50['High'] - n50['Close'].shift(1)),
                                  abs(n50['Low'] - n50['Close'].shift(1))))
n50['ATR'] = n50['TR'].rolling(window=14).mean()

# Target: Next day's movement
n50['Target'] = np.where(n50['Close'].shift(-1) > n50['Close'], 1, 0)

# Drop any rows with missing values
n50.dropna(inplace=True)

# Show the DataFrame (optional)
st.write(n50.tail())

# Plot the correlation heatmap
st.subheader('Correlation Heatmap')
plt.figure(figsize=(12, 8))
sns.heatmap(n50.corr(), annot=True, fmt=".2f", cmap="coolwarm")
st.pyplot(plt)

# Create features and target for LSTM
feature_columns = ['Close', 'Daily Return', '10ma', '50ma', '100ma', '200ma', 'Volatility', 'RSI', '20-Day SMA',
                   '20-Day StdDev', 'Upper Band', 'Lower Band', '12-Day EMA', '26-Day EMA', 'MACD', 'Signal Line',
                   'TR', 'ATR']

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(n50[feature_columns].values)

# Create sequences for LSTM


def create_sequences(data, window_size=60):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data)

# Train-test split
train_size = int(len(X) * 0.8)  # 80% for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True,
          input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_test, y_test), verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Plot actual vs predicted prices
st.subheader('LSTM Prediction vs Actual Prices')
plt.figure(figsize=(14, 8))
plt.plot(y_test, label="Actual Prices")
plt.plot(predictions, label="Predicted Prices")
plt.legend()
st.pyplot(plt)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
st.write(f"Mean Squared Error: {mse}")
st.write(f"Mean Absolute Error: {mae}")

# Fetch news and perform sentiment analysis
api_key = '09cf303023d0484383e95914664141eb'  # Replace with your API key
news_titles = fetch_news("Nifty 50", api_key)

if news_titles:
    st.subheader('Latest News')
    for i, title in enumerate(news_titles, 1):
        st.write(f"{i}. {title}")

    sentiment_scores = analyze_sentiment(news_titles)
    st.subheader('Sentiment Scores')
    for i, score in enumerate(sentiment_scores, 1):
        sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        st.write(f"{i}. Score: {score:.2f} ({sentiment})")

    # Aggregate sentiment (mean score)
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    overall_sentiment = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
    st.write(f"\nOverall Sentiment: {
             overall_sentiment} (Avg Score: {avg_sentiment:.2f})")

    # Final prediction combining price movement and sentiment
    price_movement = "Up" if y_test[-1] == 1 else "Down"
    final_prediction = "Bullish" if price_movement == "Up" and avg_sentiment > 0 else \
                       "Bearish" if price_movement == "Down" and avg_sentiment < 0 else "Uncertain"

    st.subheader('Final Prediction')
    st.write(f"\nOverall Sentiment: {
             overall_sentiment} (Avg Score: {avg_sentiment:.2f})")
    st.write(f"LSTM Prediction: {price_movement}")
    st.write(f"Final Prediction: {final_prediction}")
else:
    st.write("No news found.")
