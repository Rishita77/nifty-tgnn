import yfinance as yf
import pandas as pd
import os

TICKERS = ['SBIN.NS', 'ICICIBANK.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'ITC.NS']
os.makedirs("data/stocks_historical", exist_ok=True)

stock_data = {}
for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start='2003-01-01', end='2020-12-31', 
                     progress=False, auto_adjust=True)
    
    # Flatten multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.ffill().dropna()
    df.index = pd.to_datetime(df.index)
    
    stock_data[ticker] = df
    df.to_csv(f"data/stocks_historical/{ticker.replace('.NS','')}.csv")
    print(f"  {ticker}: {len(df)} trading days, "
          f"{df.index[0].date()} to {df.index[-1].date()}")