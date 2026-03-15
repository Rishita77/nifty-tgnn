import torch, json
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class NewsGraphBuilder:
    def __init__(self, sim_threshold=0.5):
        self.sim_threshold = sim_threshold
        
    def build_daily_news_graph(self, embeddings, sentiments, articles):
        assert len(articles) == len(sentiments) == embeddings.shape[0], \
            "Mismatched input lengths"

        N = embeddings.shape[0]
        sent_scores = torch.tensor(
            [s.get('score', 0.0) for s in sentiments], dtype=torch.float
        ).unsqueeze(1)
        x = torch.cat([embeddings, sent_scores], dim=-1)

        emb_np = embeddings.cpu().numpy()
        sim_matrix = cosine_similarity(emb_np)

        edge_index, edge_attr = [], []

        for i in range(N):
            for j in range(i + 1, N):
                sim = float(sim_matrix[i][j])
                shared = set(articles[i].get('matched_entities', [])) & \
                        set(articles[j].get('matched_entities', []))
                
                if sim > self.sim_threshold or shared:
                    weight = sim + 0.2 * len(shared)
                    edge_index += [[i, j], [j, i]]
                    edge_attr += [weight, weight]

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    


class CompanyGraphBuilder:

    SUPPLY_CHAIN = [
        ('RELIANCE.NS','ONGC.NS'),
        ('TATAMOTORS.NS','TATASTEEL.NS'),
        ('MARUTI.NS','TATASTEEL.NS'),
        ('INFY.NS','TCS.NS')
    ]

    def __init__(self, companies_path="config/nifty50_tickers.json", stock_data_path="data/processed/allstock.csv"):
        with open(companies_path) as f:
            self.companies_data = json.load(f)
        self.all_sectors = self.companies_data['sectors']

        with open(stock_data_path) as f:
            self.stock_data = pd.read_csv(f)
        
        self.stock_data['Price'] = pd.to_datetime(self.stock_data['Price'])

        self.tickers = []
        self.ticker_info_map = {}
        for company_info in self.companies_data['tickers']:
          self.tickers.append(company_info['symbol'])
          self.ticker_info_map[company_info['symbol']] = company_info
        self.ticker_to_idx = {ticker_symbol: idx for idx, ticker_symbol in enumerate(self.tickers)}

    def build_company_graph(self, date_str, lookback=20):
        date = pd.to_datetime(date_str) 
        N = len(self.tickers)
        print(f"Building graph for {N} companies on {date_str} with lookback of {lookback} days")

        sector_onehot = torch.zeros(N, len(self.all_sectors))
        print(f"Initialized sector one-hot encoding with shape: {sector_onehot.shape}")
        returns_5d = torch.zeros(N, 1)
        print(f"Initialized 5-day returns tensor with shape: {returns_5d.shape}")
        volatility = torch.zeros(N, 1)
        print(f"Initialized volatility tensor with shape: {volatility.shape}")
        
        print("Processing each company for features...")
        print(f"Tickers: {self.tickers}")

        for i, ticker_symbol in enumerate(self.tickers):
            info = self.ticker_info_map[ticker_symbol] # Corrected way to get info
            sector_idx = self.all_sectors.index(info['sector']) if info['sector'] in self.all_sectors else -1
            if sector_idx != -1: # Only set if sector exists
                sector_onehot[i, sector_idx] = 1.0

            df_ticker = self.stock_data[self.stock_data['ticker'] == ticker_symbol]
            mask = df_ticker['Price'] <= date
            recent = df_ticker[mask].tail(lookback)

            if len(recent) >= 5:
                # Calculate 5-day return (price change over 5 trading days)
                returns_5d[i] = (recent['Close'].iloc[-1] / recent['Close'].iloc[-5]) - 1
                volatility[i] = recent['Close'].pct_change().std() # Daily volatility

        x = torch.cat([sector_onehot, returns_5d, volatility], dim=-1)

        edges, weights = [], []

        for i in range(N):
            for j in range(i + 1, N):
                si = self.ticker_info_map[self.tickers[i]]['sector'] # Corrected
                sj = self.ticker_info_map[self.tickers[j]]['sector'] # Corrected
                if si == sj:
                    edges.extend([[i, j], [j, i]])
                    weights.extend([1.0, 1.0])

        for t1, t2 in self.SUPPLY_CHAIN:
            if t1 in self.ticker_to_idx and t2 in self.ticker_to_idx:
                i, j = self.ticker_to_idx[t1], self.ticker_to_idx[t2]
                # Avoid adding duplicate edges if already added by same-sector logic,
                # or ensure higher weight takes precedence.
                # For simplicity here, just add them. Graph will handle duplicates later.
                edges.extend([[i, j], [j, i]])
                weights.extend([0.7, 0.7])

        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float) if weights else torch.zeros(0)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

if __name__ == "__main__":
    # builder = NewsGraphBuilder(sim_threshold=0.5)
    # embeddings = torch.load("data/processed/embeddings/embeddings_2024-01-01 (1).pt")

    # # load articles
    # with open("data/processed_articles4.json") as f:
    #     articles = json.load(f)

    # # load sentiments
    # with open("data/processed_articles_with_sentiment_c.json") as f:
    #     sentiments = json.load(f)

    # # check
    # # assert len(articles) == len(sentiments) == embeddings.shape[0]
    # embedding_slice = embeddings[0:10]
    # graph_data = builder.build_daily_news_graph(embedding_slice, sentiments[0:10], articles[0:10])
    # print(graph_data)
    
    # company_config_file = "config/nifty50_tickers.json"
    # company_config = json.load(open(company_config_file))
    
    company_builder = CompanyGraphBuilder()
    company_graph = company_builder.build_company_graph(date_str="2024-01-01")
    print(company_graph)