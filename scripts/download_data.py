import yfinance as yf
import pandas as pd
import numpy as np
import json, time, os

def download_nifty50_fixed(config_path="config/nifty50_tickers.json",
                          start="2003-04-01", end="2020-05-26",
                          threshold_pct=1.5, window_days=5):
    with open(config_path) as f:
        data = json.load(f)
        companies = [item['symbol'] for item in data['tickers']]
    
    all_data = {}
    for ticker in companies:
        print(f"Processing {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[['Open','High','Low','Close','Volume']].copy()
        df = df.ffill().dropna()
        
        # FIXED LABELS
        print(ticker)
        df = create_real_signal_labels(df, window_days=window_days, z_threshold=threshold_pct)
        
        os.makedirs("data/raw/stocks_zscore", exist_ok=True)
        df.to_csv(f"data/raw/stocks_zscore/{ticker.replace('.NS','')}.csv")
        all_data[ticker] = df
    
    return all_data

def create_real_signal_labels(df, window_days=5, z_threshold=1.5):
    """Label only statistically significant moves relative to volatility"""
    df = df.copy()
    
    # 5-day forward return
    df['fwd_return'] = df['Close'].pct_change(window_days).shift(-window_days)
    
    # Rolling volatility (normalizes across stocks)
    df['vol_20d'] = df['Close'].pct_change().rolling(20).std()
    df['z_score'] = df['fwd_return'] / df['vol_20d']
    
    # Only label moves >1.5σ (top ~13% extreme moves)
    df['direction'] = 0
    df['direction'] = np.where(df['z_score'] > z_threshold, 1, df['direction'])
    df['direction'] = np.where(df['z_score'] < -z_threshold, -1, df['direction'])
    
    signal_df = df[df['direction'] != 0].dropna()
    
    print(f"Signal days: {len(signal_df):,} ({len(signal_df)/len(df)*100:.1f}%)")
    print(f"UP/DOWN: {signal_df['direction'].value_counts(normalize=True)}")
    print(f"Z-score thresholds: ±{z_threshold}")
    
    return signal_df

# Replace your download function body with:
# df = yf.download(ticker, start=start, end=end, progress=False)
# df = df[['Open','High','Low','Close','Volume']].ffill().dropna()
# df = create_signal_labels(df)

if __name__ == "__main__":
    data = download_nifty50_fixed()