import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Google Sheets API configuration
SERVICE_ACCOUNT_FILE = r"C:\Users\rohan\Downloads\dark-valor-426921-b2-03329f14e050.json"
SHEET_NAME = 'Stock Price Data - Technology'

# Function to fetch historical stock data for multiple symbols
def fetch_historical_data(symbols, start_date, end_date):
    data = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)
        df['Symbol'] = symbol  # Add symbol as a column for identification
        data[symbol] = df.copy()  # Ensure to make a copy of the DataFrame

    return data

# Function to calculate additional technical indicators
def calculate_technical_indicators(data):
    for symbol, df in data.items():
        # Simple Moving Average (SMA)
        df.loc[:, 'SMA_20'] = df['Close'].rolling(window=20).mean()

        # Exponential Moving Average (EMA)
        df.loc[:, 'EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # Volatility (Standard Deviation)
        df.loc[:, 'Volatility_20'] = df['Close'].rolling(window=20).std()

        # Bollinger Bands
        df.loc[:, 'BB_upper'] = df['SMA_20'] + 2 * df['Volatility_20']
        df.loc[:, 'BB_lower'] = df['SMA_20'] - 2 * df['Volatility_20']

        # Average True Range (ATR)
        df.loc[:, 'High-Low'] = df['High'] - df['Low']
        df.loc[:, 'High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df.loc[:, 'Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df.loc[:, 'TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        df.loc[:, 'ATR_14'] = df['TR'].rolling(window=14).mean()

        # Relative Strength Index (RSI)
        df.loc[:, 'Change'] = df['Close'].diff()
        df.loc[:, 'Gain'] = df['Change'].apply(lambda x: x if x > 0 else 0)
        df.loc[:, 'Loss'] = df['Change'].apply(lambda x: -x if x < 0 else 0)
        df.loc[:, 'Avg_Gain'] = df['Gain'].rolling(window=14).mean()
        df.loc[:, 'Avg_Loss'] = df['Loss'].rolling(window=14).mean()
        df.loc[:, 'RS'] = df['Avg_Gain'] / df['Avg_Loss']
        df.loc[:, 'RSI'] = 100 - (100 / (1 + df['RS']))

    return data

# Function to update Google Sheets with stock data including technical indicators
def update_google_sheet(data, sheet_name):
    SCOPES = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)

    try:
        sheet = client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        print(f"Spreadsheet '{sheet_name}' not found. Please ensure the name is correct and the service account has access.")
        return

    for symbol, df in data.items():
        worksheet_name = f'{symbol}'
        try:
            worksheet = sheet.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            print(f"Worksheet '{worksheet_name}' not found, creating a new one.")
            worksheet = sheet.add_worksheet(title=worksheet_name, rows="1000", cols="20")

        try:
            # Reverse the order of rows (most recent first)
            df = df.iloc[::-1].copy()  # Ensure a copy is made and reversed

            # Convert the index to a column and ensure all data is in a serializable format
            df.reset_index(inplace=True)
            df['Date'] = df['Date'].astype(str)  # Ensure the date is a string

            # Calculate additional technical indicators
            df = calculate_technical_indicators({symbol: df})[symbol]

            # Convert all Timestamp objects to string
            df = df.astype(str)

            worksheet.clear()
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
            print(f"Updated worksheet '{worksheet_name}' with {len(df)} rows including technical indicators.")
        except Exception as e:
            print(f"Error updating worksheet '{worksheet_name}': {e}")

# Example usage
symbols = ['AAPL']  # List of symbols to fetch data for
start_date = '2014-01-01'
end_date = '2024-08-14'

data = fetch_historical_data(symbols, start_date, end_date)
update_google_sheet(data, SHEET_NAME)

print("Google Sheet updated successfully with additional technical indicators.")

def fetch_historical_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    df['Symbol'] = symbol  # Add symbol as a column for identification
    return df

# Function to prepare data with technical indicators
def prepare_data(df):
    # Create technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()

    # Create lagged returns
    df['Return_1d'] = df['Close'].pct_change()
    df['Return_5d'] = df['Close'].pct_change(5)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Define features and target
    X = df[['SMA_20', 'EMA_20', 'Volatility_20', 'Return_1d', 'Return_5d']]
    y = df['Close']

    return X, y

# Function to train and evaluate the model
def train_and_evaluate(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model using Mean Absolute Error (MAE)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    print(f"Mean Absolute Error (MAE) on training set: {mae_train}")
    print(f"Mean Absolute Error (MAE) on testing set: {mae_test}")

    return model

# Function to make future predictions
def predict_future_prices(model, X_future):
    return model.predict(X_future)

# Example usage
symbol = 'AAPL'  # Symbol to fetch data for

# Fetch historical data
df = fetch_historical_data(symbol, start_date, end_date)

# Prepare data
X, y = prepare_data(df)

# Train and evaluate the model
model = train_and_evaluate(X, y)

# Prepare future data for predictions (assuming we have future features)
X_future = X.tail(1)  # Use the latest available data as an example
predicted_price = predict_future_prices(model, X_future)
print(f"Predicted future price: {predicted_price[0]}")
