# news_aggregation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData


# ═══════════════════════════════════════════════════════════════
# Three aggregation strategies
# ═══════════════════════════════════════════════════════════════

class MeanPoolingAggregator(nn.Module):
    """
    (a) Plain mean of all news embeddings mentioning a company.
    No parameters — fast baseline.
    """
    def forward(self, news_x, edge_index, edge_attr, num_companies):
        # edge_index: (2, E)  row0=news_idx, row1=company_idx
        src, dst = edge_index                          # src=news, dst=company
        out = torch.zeros(num_companies, news_x.size(1), device=news_x.device)
        cnt = torch.zeros(num_companies, 1,              device=news_x.device)

        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, news_x.size(1)), news_x[src])
        cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.size(0), 1, device=news_x.device))

        cnt = cnt.clamp(min=1)          # avoid div-by-zero for unmentioned companies
        return out / cnt                # (N_companies, hidden)


class AttentionPoolingAggregator(nn.Module):
    """
    (b) Learned attention: each company learns to weight the news
        that mentions it. The attention score is computed from
        both the news embedding and the company embedding.

    score_i = v^T tanh(W_n * news_i + W_c * company + b)
    """
    def __init__(self, hidden):
        super().__init__()
        self.W_news    = nn.Linear(hidden, hidden, bias=False)
        self.W_company = nn.Linear(hidden, hidden, bias=False)
        self.v         = nn.Linear(hidden, 1,      bias=False)

    def forward(self, news_x, company_x, edge_index, num_companies):
        src, dst = edge_index           # src=news idx, dst=company idx

        if src.numel() == 0:
            return torch.zeros(num_companies, news_x.size(1), device=news_x.device)

        # Pairwise score for each (news, company) mention edge
        news_part    = self.W_news(news_x[src])        # (E, hidden)
        company_part = self.W_company(company_x[dst])  # (E, hidden)
        scores       = self.v(torch.tanh(news_part + company_part)).squeeze(-1)  # (E,)

        # Softmax per company — each company normalises over its own mentions
        # We use a scatter softmax manually
        scores_exp = scores - scatter_max(scores, dst, num_companies)  # numerical stability
        scores_exp = scores_exp.exp()
        denom      = scatter_sum(scores_exp, dst, num_companies)[dst].clamp(min=1e-9)
        weights    = scores_exp / denom                # (E,) — softmax per company

        # Weighted sum
        out = torch.zeros(num_companies, news_x.size(1), device=news_x.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, news_x.size(1)),
            news_x[src] * weights.unsqueeze(1)
        )
        return out                                     # (N_companies, hidden)


class SentimentPoolingAggregator(nn.Module):
    """
    (c) Weight each news embedding by abs(sentiment_score).
    edge_attr holds the pre-computed abs(sentiment) + 0.1 weights
    from build_combined_graph.
    No learned parameters.
    """
    def forward(self, news_x, edge_index, edge_attr, num_companies):
        src, dst = edge_index

        if src.numel() == 0:
            return torch.zeros(num_companies, news_x.size(1), device=news_x.device)

        weights = edge_attr.unsqueeze(1)               # (E, 1) — sentiment weights

        # Normalise weights per company so they sum to 1
        weight_sum = torch.zeros(num_companies, 1, device=news_x.device)
        weight_sum.scatter_add_(0, dst.unsqueeze(1), weights)
        norm_weights = weights / weight_sum[dst].clamp(min=1e-9)  # (E, 1)

        out = torch.zeros(num_companies, news_x.size(1), device=news_x.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, news_x.size(1)),
            news_x[src] * norm_weights
        )
        return out                                     # (N_companies, hidden)


# ── Scatter helpers (avoid torch_scatter dependency) ──────────

def scatter_sum(src, index, num_nodes):
    out = torch.zeros(num_nodes, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, index, src)
    return out

def scatter_max(src, index, num_nodes):
    """Per-group max — for numerical stability in attention softmax."""
    out = torch.full((num_nodes,), float('-inf'), device=src.device, dtype=src.dtype)
    for i, (s, idx) in enumerate(zip(src, index)):
        if s > out[idx]:
            out[idx] = s
    # Vectorised version:
    # out, _ = torch_scatter.scatter_max(src, index, out=out)
    return out[index]   # return per-edge max of its group


# ═══════════════════════════════════════════════════════════════
# HeteroGNN with pluggable aggregation
# ═══════════════════════════════════════════════════════════════

class HeteroNewsCompanyGNN(nn.Module):
    """
    Replaces HeteroConv's built-in news→company message passing
    with one of three explicit aggregation strategies.

    aggr: 'mean' | 'attention' | 'sentiment'
    """
    def __init__(self, news_dim=385, company_dim=24, hidden=64,
                 dropout=0.3, aggr='mean'):
        super().__init__()
        assert aggr in ('mean', 'attention', 'sentiment'), \
            f"aggr must be 'mean', 'attention', or 'sentiment', got {aggr}"
        self.aggr    = aggr
        self.dropout = dropout

        # ── Projections ───────────────────────────────────────
        self.news_proj    = nn.Sequential(Linear(news_dim,    hidden), nn.ReLU())
        self.company_proj = nn.Sequential(Linear(company_dim, hidden), nn.ReLU())

        # ── Aggregators ───────────────────────────────────────
        if aggr == 'mean':
            self.news_aggregator = MeanPoolingAggregator()
        elif aggr == 'attention':
            self.news_aggregator = AttentionPoolingAggregator(hidden)
        else:
            self.news_aggregator = SentimentPoolingAggregator()

        # News message fuses aggregated news into company rep
        self.news_fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),   # [company || aggregated_news] → hidden
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        # ── Graph convolutions (company-only after news fusion) ──
        self.conv1 = HeteroConv({
            ('news',    'similar_to', 'news'):    SAGEConv((hidden, hidden), hidden),
            ('company', 'related_to', 'company'): SAGEConv((hidden, hidden), hidden),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('news',    'similar_to', 'news'):    SAGEConv((hidden, hidden), hidden),
            ('company', 'related_to', 'company'): SAGEConv((hidden, hidden), hidden),
        }, aggr='mean')

        self.norm_news    = nn.LayerNorm(hidden)
        self.norm_company = nn.LayerNorm(hidden)

        # ── Classifier ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        n_companies = data['company'].x.size(0)
        edge_index  = data['news', 'mentions', 'company'].edge_index   # (2, E)
        edge_attr   = data['news', 'mentions', 'company'].get('edge_attr',
                          torch.ones(edge_index.size(1), device=edge_index.device))

        # ── Project both node types ───────────────────────────
        news_h    = self.news_proj(data['news'].x)       # (N_news, hidden)
        company_h = self.company_proj(data['company'].x) # (N_companies, hidden)

        # ── Aggregate news → company ──────────────────────────
        if self.aggr == 'attention':
            news_agg = self.news_aggregator(
                news_h, company_h, edge_index, n_companies
            )
        else:
            news_agg = self.news_aggregator(
                news_h, edge_index, edge_attr, n_companies
            )
        # news_agg: (N_companies, hidden) — zero for unmentioned companies

        # ── Fuse news signal into company representation ──────
        company_h = self.news_fuse(
            torch.cat([company_h, news_agg], dim=-1)
        )                                                # (N_companies, hidden)

        # ── Graph convolutions over news + company graphs ─────
        x_dict = {'news': news_h, 'company': company_h}

        # Build edge_index_dict WITHOUT the mentions edge
        # (already handled above — don't double-count)
        edge_index_dict = {
            ('news',    'similar_to', 'news'):    data['news', 'similar_to', 'news'].edge_index,
            ('company', 'related_to', 'company'): data['company', 'related_to', 'company'].edge_index,
        }

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {
            'news':    self.norm_news(F.relu(x_dict['news'])),
            'company': self.norm_company(F.relu(x_dict['company'])),
        }
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                  for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        company_out = self.norm_company(F.relu(x_dict['company']))

        return self.classifier(company_out).squeeze(-1)  # (N_companies,) raw logits


# ═══════════════════════════════════════════════════════════════
# Comparison runner
# ═══════════════════════════════════════════════════════════════

def compare_aggregators(dataset, splits, news_dim=385, company_dim=24,
                        hidden=64, device='cpu', patience=10):
    """
    Trains all three aggregators on the same walk-forward splits
    and returns a comparison DataFrame.
    """
    import pandas as pd
    from trainer import Trainer, make_walk_forward_splits

    results = {}

    for aggr_name in ('mean', 'sentiment', 'attention'):
        print(f"\n{'═'*55}")
        print(f"  Aggregator: {aggr_name.upper()}")
        print(f"{'═'*55}")

        model = HeteroNewsCompanyGNN(
            news_dim=news_dim, company_dim=company_dim,
            hidden=hidden, aggr=aggr_name
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        trainer = Trainer(
            model=model, optimizer=optimizer, device=device,
            patience=patience,
            checkpoint_path=f'checkpoints/best_{aggr_name}.pt'
        )

        trainer.walk_forward_train(dataset, splits)

        df = trainer.get_history_df()

        # Best val metrics per fold
        best_per_fold = df.groupby('fold').apply(
            lambda g: g.loc[g['val_loss'].idxmin()]
        )[['val_loss', 'val_acc', 'val_f1']]

        results[aggr_name] = {
            'mean_val_loss': best_per_fold['val_loss'].mean(),
            'mean_val_acc':  best_per_fold['val_acc'].mean(),
            'mean_val_f1':   best_per_fold['val_f1'].mean(),
            'history':       df,
        }

        print(f"\n  {aggr_name} summary:")
        print(f"    Val loss : {results[aggr_name]['mean_val_loss']:.4f}")
        print(f"    Val acc  : {results[aggr_name]['mean_val_acc']:.3f}")
        print(f"    Val F1   : {results[aggr_name]['mean_val_f1']:.3f}")

    # ── Summary table ─────────────────────────────────────────
    summary = pd.DataFrame({
        k: {m: v for m, v in vals.items() if m != 'history'}
        for k, vals in results.items()
    }).T.sort_values('mean_val_f1', ascending=False)

    print(f"\n{'═'*55}")
    print("COMPARISON SUMMARY (sorted by val F1)")
    print(summary.to_string())
    print(f"{'═'*55}")

    return results, summary
