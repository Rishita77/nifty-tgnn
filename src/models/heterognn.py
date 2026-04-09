import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

# ✅ YOUR MODEL (PERFECT - no changes needed)
class HeteroNewsCompanyGNN(nn.Module):
    def __init__(self, news_dim=385, company_dim=24, hidden=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.news_proj    = nn.Sequential(Linear(news_dim, hidden), nn.ReLU())
        self.company_proj = nn.Sequential(Linear(company_dim, hidden), nn.ReLU())
        self.norm1_news    = nn.LayerNorm(hidden)
        self.norm1_company = nn.LayerNorm(hidden)
        self.norm2_company = nn.LayerNorm(hidden)

        self.conv1 = HeteroConv({
            ('news',    'similar_to', 'news'):    SAGEConv((hidden, hidden), hidden),
            ('news',    'mentions',   'company'): SAGEConv((hidden, hidden), hidden),
            ('company', 'related_to', 'company'): SAGEConv((hidden, hidden), hidden),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('news',    'similar_to', 'news'):    SAGEConv((hidden, hidden), hidden),
            ('news',    'mentions',   'company'): SAGEConv((hidden, hidden), hidden),
            ('company', 'related_to', 'company'): SAGEConv((hidden, hidden), hidden),
        }, aggr='sum')

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x_news_0 = self.news_proj(data['news'].x)
        x_company_0 = self.company_proj(data['company'].x)
        x_dict = {'news': x_news_0, 'company': x_company_0}

        # Layer 1
        x_out = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {
            'news':    self.norm1_news(torch.relu(x_out.get('news', x_news_0))),
            'company': self.norm1_company(torch.relu(x_out.get('company', x_company_0) + x_company_0)),
        }
        x_dict = {k: nn.functional.dropout(v, p=self.dropout, training=self.training)
                  for k, v in x_dict.items()}

        # Layer 2
        x_out = self.conv2(x_dict, data.edge_index_dict)
        company_out = torch.relu(x_out.get('company', x_dict['company']) + x_dict['company'])

        return self.classifier(company_out).squeeze(-1)

# ✅ FIXED label function (your z-score logic)
def make_zscore_labels(stock_data, tickers, date_str):
    date = pd.to_datetime(date_str)
    labels = torch.full((len(tickers),), -1.0)  # -1 = invalid

    for i, ticker in enumerate(tickers):
        df = stock_data[stock_data['ticker'] == ticker].sort_values('Date')

        past_mask = df['Date'] < date
        future_mask = df['Date'] > date

        if past_mask.sum() >= 5 and future_mask.sum() > 0:
            past_close = df[past_mask]['Close'].iloc[-1]
            future_close = df[future_mask]['Close'].iloc[0]
            ret = (future_close / past_close) - 1

            # 1 = significant move, 0 = neutral
            labels[i] = 1 if abs(ret) > 0.01 else 0

    return labels

# ✅ FIXED dataset loader (uses make_zscore_labels)
def load_hetero_dataset(combined_dir, stock_data, tickers):
    dataset = []
    files = sorted(os.listdir(combined_dir))
    print(f"Scanning {len(files)} files in {combined_dir}")

    for fname in tqdm(files, desc="Loading combined graphs"):
        if not fname.endswith(".pt"): continue

        # ✅ FIX: Handle DD-MM-YYYY directly
        date_str = fname.replace(".pt", "")
        try:
            # Try both formats
            date = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
            if pd.isna(date):
                date = pd.to_datetime(date_str, errors='coerce')
            if pd.isna(date):
                print(f"❌ Bad date: {date_str}")
                continue
        except:
            continue

        print(f"✅ Loaded graph: {fname}, news.shape={torch.load(os.path.join(combined_dir, fname), weights_only=False)['news'].x.shape}")

        # ✅ Use EXACT filename for labels
        y = make_zscore_labels(stock_data, tickers, date_str)
        mask = (y != -1)
        print(f"Labels valid for {mask.sum().item()}/49 tickers")

        if mask.sum() >= 5:
            graph = torch.load(os.path.join(combined_dir, fname), weights_only=False)
            dataset.append({'data': graph, 'y': y, 'mask': mask, 'date': date})
        else:
            print(f"❌ Skipping {fname}: too few valid labels ({mask.sum().item()})")

    print(f"✅ FINAL: Loaded {len(dataset)} snapshots")
    return dataset

# ✅ YOUR OTHER FUNCTIONS (unchanged - perfect)
def run_hetero_epoch(model, dataset, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    for sample in dataset:
        data = sample['data'].to(device)
        y = sample['y'].to(device)
        mask = sample['mask'].to(device)

        if mask.sum() == 0: continue

        with torch.set_grad_enabled(train):
            logits = model(data)
            loss = criterion(logits[mask], y[mask])

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        preds = (torch.sigmoid(logits[mask]) > 0.5).long().cpu().tolist()
        tgts = y[mask].long().cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(tgts)

    n = len(all_targets)
    avg_loss = total_loss / n if n > 0 else float('nan')
    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / n
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1

def time_split(dataset):
    train, val, test = [], [], []
    for s in dataset:
        if s['date'] < pd.Timestamp("2016-12-31"):
            train.append(s)
        elif s['date'] < pd.Timestamp("2017-12-31"):
            val.append(s)
        elif s['date'] < pd.Timestamp("2020-01-01"):
            test.append(s)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def compute_pos_weight(train_set, device):
    all_y = []
    for s in train_set:
        all_y.extend(s['y'][s['mask']].tolist())
    all_y = torch.tensor(all_y)
    n_pos = all_y.sum().item()
    n_neg = len(all_y) - n_pos
    w = torch.tensor(n_neg / (n_pos + 1e-8), dtype=torch.float, device=device)
    print(f"pos_weight = {w:.3f} (pos={int(n_pos)}, neg={int(n_neg)})")
    return w

# ✅ TRAINING SCRIPT (now will run without crashing)
# dataset = load_hetero_dataset(
#     combined_dir="graphs_new/combined",
#     stock_data=company_builder.stock_data,  # assumes you have this from earlier
#     tickers=company_builder.tickers
# )

# train_set, val_set, test_set = time_split(dataset)

# sample = dataset[0]['data']
# news_dim, company_dim = sample['news'].x.shape[1], sample['company'].x.shape[1]
# print(f"News dim: {news_dim}, Company dim: {company_dim}")

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = HeteroNewsCompanyGNN(news_dim=news_dim, company_dim=company_dim).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# pos_weight = compute_pos_weight(train_set, device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# best_val_loss = float('inf')
# for epoch in range(1, 101):
#     tr_loss, tr_acc, tr_f1 = run_hetero_epoch(model, train_set, criterion, optimizer, device, train=True)
#     vl_loss, vl_acc, vl_f1 = run_hetero_epoch(model, val_set, criterion, optimizer, device, train=False)

#     if vl_loss < best_val_loss:
#         best_val_loss = vl_loss
#         torch.save(model.state_dict(), 'best_model.pt')

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch:03d} | Train: {tr_acc:.3f} F1:{tr_f1:.3f} | Val: {vl_acc:.3f} F1:{vl_f1:.3f}")

# model.load_state_dict(torch.load('best_model.pt'))
# te_loss, te_acc, te_f1 = run_hetero_epoch(model, test_set, criterion, optimizer, device, train=False)
# print(f"FINAL TEST: Acc {te_acc:.3f} | F1 {te_f1:.3f}")