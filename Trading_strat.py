import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbols
tickerSymbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN','TSLA']

# Initialize a DataFrame to store the results
results = pd.DataFrame()

# Loop over the ticker symbols
for tickerSymbol in tickerSymbols:

    # Get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)

    # Get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2016-1-1', end='2023-6-18')

    # Calculate the 10 day moving average
    tickerDf['MA10'] = tickerDf['Close'].rolling(window=10).mean()

    # Create a column 'Shares' where if MA10 is greater than its previous day then set to 1 (long one share of stock), else 0 (exit position)
    tickerDf['Shares'] = [1 if ma > ma_shifted else 0 for ma, ma_shifted in zip(tickerDf['MA10'], tickerDf['MA10'].shift(1))]

    # Add a column 'Close1' that aids in calculating the daily returns from this strategy
    tickerDf['Close1'] = tickerDf['Close'].shift(-1)

    # Add a column 'Profit' which is the daily profit (or loss)
    tickerDf['Profit'] = [tickerDf.loc[ei, 'Close1'] - tickerDf.loc[ei, 'Close'] if tickerDf.loc[ei, 'Shares']==1 else 0 for ei in tickerDf.index]

    # Calculate the cumulative profit and add it to the results DataFrame
    results[tickerSymbol] = tickerDf['Profit'].cumsum()

# Plot the results
results.plot()
plt.show()
