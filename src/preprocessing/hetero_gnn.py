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