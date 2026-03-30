import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class StaticGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, 
                 num_layers=2, gnn_type='gcn', dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        Conv = GCNConv if gnn_type == 'gcn' else GATConv
        
        # First layer
        self.convs.append(Conv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(Conv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Final layer
        self.convs.append(Conv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_channels),
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        
        out = self.classifier(x)  # (N, 1)
        return torch.sigmoid(out).squeeze(-1)