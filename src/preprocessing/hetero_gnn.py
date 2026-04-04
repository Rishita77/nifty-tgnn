import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData

class HeteroNewsCompanyGNN(nn.Module):
    def __init__(self, news_dim=385, company_dim=24, hidden=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Project both node types to common dimension
        self.news_proj    = nn.Sequential(Linear(news_dim, hidden), nn.ReLU())
        self.company_proj = nn.Sequential(Linear(company_dim, hidden), nn.ReLU())

        # Layer norms — safer than BatchNorm for small graphs
        self.norm1_news    = nn.LayerNorm(hidden)
        self.norm1_company = nn.LayerNorm(hidden)
        self.norm2_company = nn.LayerNorm(hidden)

        # SAGEConv works cleanly in HeteroConv for all edge types
        self.conv1 = HeteroConv({
            ('news',    'similar_to', 'news'):    SAGEConv((hidden, hidden), hidden),
            ('news',    'mentions',   'company'): SAGEConv((hidden, hidden), hidden),
            ('company', 'related_to', 'company'): SAGEConv((hidden, hidden), hidden),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('news',    'similar_to', 'news'):    SAGEConv((hidden, hidden), hidden),
            ('news',    'mentions',   'company'): SAGEConv((hidden, hidden), hidden),
            ('company', 'related_to', 'company'): SAGEConv((hidden, hidden), hidden),
        }, aggr='mean')

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        # Project to hidden dim
        x_dict = {
            'news':    self.news_proj(data['news'].x),
            'company': self.company_proj(data['company'].x),
        }

        # Conv layer 1
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            'news':    self.norm1_news(torch.relu(x_dict['news'])),
            'company': self.norm1_company(torch.relu(x_dict['company'])),
        }
        x_dict = {k: nn.functional.dropout(v, p=self.dropout, training=self.training)
                  for k, v in x_dict.items()}

        # Conv layer 2
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict['company'] = self.norm2_company(torch.relu(x_dict['company']))

        # Raw logits — no sigmoid here, BCEWithLogitsLoss handles it
        return self.classifier(x_dict['company']).squeeze(-1)  # (N_companies,)
    
    
    
    
    # ── Usage ──

# 1. Define your ticker mapping
import json
with open("config/nifty50_tickers.json") as f:
    companies = json.load(f)
tickers = sorted(companies.keys())
ticker_to_idx = {t: i for i, t in enumerate(tickers)}

# 2. Combine graphs (run once)
combine_daily_graphs(
    company_dir="graphs/company",
    news_dir="graphs",
    output_dir="graphs/combined",
    articles_dir="data/processed/news_embeddings",
    ticker_to_idx=ticker_to_idx,
)

# 3. Load combined dataset
dataset = load_hetero_dataset(
    combined_dir="graphs/combined",
    stock_data=company_builder.stock_data,
    tickers=tickers,
)

# 4. Split and train
train_set, val_set, test_set = time_split(dataset)

# Check dimensions from first graph
sample = dataset[0]['data']
news_dim = sample['news'].x.shape[1]
company_dim = sample['company'].x.shape[1]
print(f"News features: {news_dim}, Company features: {company_dim}")

model = HeteroNewsCompanyGNN(
    news_dim=news_dim,
    company_dim=company_dim,
    hidden=64,
    dropout=0.3,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
pos_weight = compute_pos_weight(train_set, device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Same training loop as before, just use run_hetero_epoch
best_val_loss, wait, best_state = float('inf'), 0, None

for epoch in range(1, 101):
    tr_loss, tr_acc, tr_f1 = run_hetero_epoch(model, train_set, criterion, optimizer, device, train=True)
    vl_loss, vl_acc, vl_f1 = run_hetero_epoch(model, val_set, criterion, optimizer, device, train=False)

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= 10:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"Val loss {vl_loss:.4f} acc {vl_acc:.3f} f1 {vl_f1:.3f}")

model.load_state_dict(best_state)
te_loss, te_acc, te_f1 = run_hetero_epoch(model, test_set, criterion, optimizer, device, train=False)
print(f"\nTest loss {te_loss:.4f} | acc {te_acc:.3f} | f1 {te_f1:.3f}")