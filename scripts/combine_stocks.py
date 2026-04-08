import pandas as pd
import glob
import os
import numpy as np


def load_stock_csv(path):
    sample = pd.read_csv(path, header=None, nrows=10, dtype=str)
    skiprows = []
    if len(sample) > 1 and str(sample.iloc[1, 0]).strip().lower() == 'ticker':
        skiprows.append(1)
    if len(sample) > 2 and str(sample.iloc[2, 0]).strip().lower() == 'date':
        skiprows.append(2)

    df = pd.read_csv(path, header=None, skiprows=skiprows, dtype=str)
    first_row = df.iloc[0].tolist()
    header_tokens = [str(x).strip().lower() for x in first_row if pd.notna(x) and str(x).strip()]
    second_row_is_date = False
    if df.shape[0] > 1:
        second_value = str(df.iloc[1, 0]).strip()
        second_row_is_date = pd.to_datetime(second_value, errors='coerce') is not pd.NaT

    if header_tokens and any(tok in header_tokens for tok in ['price', 'open', 'high', 'low', 'close', 'volume', 'fwd_return', 'direction']) and second_row_is_date:
        if first_row and str(first_row[0]).strip().lower() == 'price':
            columns = ['Date'] + [str(x).strip() for x in first_row[1:]]
        else:
            columns = [str(x).strip() for x in first_row]
        df = df.iloc[1:].copy()
        df.columns = columns
    else:
        df.columns = [f'col_{i}' for i in range(df.shape[1])]

    return df


if __name__ == "__main__":
    # NUCLEAR OPTION - rebuild from individual CSVs with FORCE date parsing
    files = glob.glob("data/raw/stocks_zscore/*.csv")
    dfs = []

    for file in files:
        ticker = os.path.basename(file).replace('.csv', '')
        df = load_stock_csv(file)

        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isna().all():
            raise ValueError(f"Unable to parse a date column for file {file}")

        df = df.set_index(date_col)
        df.index.name = 'Date'

        df['ticker'] = ticker
        if 'direction' not in df.columns:
            raise ValueError(f"Missing 'direction' column in file {file}")

        df = df[df['direction'] != 0].copy()
        dfs.append(df)
        print(f"✅ {ticker}: {len(df)} rows, index OK")

    allstocks = pd.concat(dfs, ignore_index=False)
    allstocks.to_csv("data/processed/allstocks_signal_FIXED.csv")

    print("\n🎉 SUCCESS CHECK")
    print(f"Index type: {type(allstocks.index)}")
    print(f"Index sample: {allstocks.index[:5]}")
    print(f"Shape: {allstocks.shape}")
    print(allstocks[['ticker', 'Close', 'direction']].head(10))