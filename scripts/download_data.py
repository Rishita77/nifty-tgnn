"""Download all NIFTY 50 historical stock data using yfinance."""
import json
import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nifty50_data(
    tickers_path: str = "config/nifty50_tickers.json",
    output_dir: str = "data/raw/stocks",
    start: str = "2025-02-07",
    end: str = "2025-08-21"
) -> dict:
    """
    Download OHLCV data for all NIFTY 50 stocks.
    Returns dict of {ticker: DataFrame}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(tickers_path) as f:
        config = json.load(f)
    
    results = {}
    failed = []
    
    for stock in config["tickers"]:
        symbol = stock["symbol"]
        try:
            logger.info(f"Downloading {symbol}...")
            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True 
            )
            # if len(df) < 100:
            #     logger.warning(f"{symbol}: Only {len(df)} rows — skipping")
            #     failed.append(symbol)
            #     continue
            
            df["ticker"] = symbol
            df["sector"] = stock["sector"]
            df["name"] = stock["name"]
            
            out_path = Path(output_dir) / f"{symbol.replace('.NS', '')}.csv"
            df.to_csv(out_path)
            results[symbol] = df
            logger.info(f"  ✓ {symbol}: {len(df)} rows saved to {out_path}")
            
        except Exception as e:
            logger.error(f"  ✗ {symbol}: {e}")
            failed.append(symbol)
    
    logger.info(f"\nDownloaded: {len(results)}/{len(config['tickers'])} stocks")
    if failed:
        logger.warning(f"Failed: {failed}")
    
    return results

def preprocess_data(df : pd.DataFrame) -> pd.DataFrame:
    """Removes redundant rows"""
    
    df = df.drop(index=[0, 1]) # Ticker metadata row
    return df

if __name__ == "__main__":
    data = download_nifty50_data()
    print(f"\n Data pipeline ready: {len(data)} stocks downloaded")
    
    path_dir = Path("data/raw/stocks")
    for csv_file in path_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        df = preprocess_data(df)
        df.to_csv(csv_file)