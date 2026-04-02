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
    
    
    
# ─────────────────────────────────────────────
# Step 4 — Training loop
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.optim import Adam

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model     = StaticGNN(in_channels=24, hidden_channels=64, num_layers=2).to(device)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()   # handles sigmoid internally — safer than BCE + manual sigmoid

def run_epoch(dataset, train=True):
    model.train() if train else model.eval()
    total_loss, total_correct, total_nodes = 0, 0, 0

    for sample in dataset:
        data = sample['data'].to(device)
        y    = sample['y'].to(device)
        mask = sample['mask'].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out  = model(data)               # (50,) raw logits
            loss = criterion(out[mask], y[mask])

        if train:
            loss.backward()
            optimizer.step()

        # Accuracy
        preds   = (torch.sigmoid(out[mask]) > 0.5).float()
        correct = (preds == y[mask]).sum().item()

        total_loss   += loss.item() * mask.sum().item()
        total_correct += correct
        total_nodes   += mask.sum().item()

    avg_loss = total_loss / total_nodes
    accuracy = total_correct / total_nodes
    return avg_loss, accuracy


# ── Training ────────────────────────────────
NUM_EPOCHS = 30
best_val_loss = float('inf')
best_state    = None

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_set, train=True)
    val_loss,   val_acc   = run_epoch(val_set,   train=False)

    # Save best model by val loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
              f"Val loss:   {val_loss:.4f} acc: {val_acc:.3f}")

# ── Test ────────────────────────────────────
model.load_state_dict(best_state)
test_loss, test_acc = run_epoch(test_set, train=False)
print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.3f}")