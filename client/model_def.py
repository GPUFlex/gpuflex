import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

class RealEstateModel(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.block = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.block(x)
        return self.output_layer(x)


def train_model(model, data, device='cpu'):
    x, y = data
    x, y = x.to(device), y.to(device)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in trange(30, desc="Training"):
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

    return loss.item()