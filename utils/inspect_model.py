import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# === Define the model class ===
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

# === Load model state_dict ===
model_path = "test.pth"

if not os.path.exists(model_path):
    print(f"❌ File {model_path} not found.")
    exit(1)

state_dict = torch.load(model_path, map_location="cpu")
print(f"📦 Loaded model state_dict with {len(state_dict)} keys")

# === Rebuild model and load weights ===
model = RealEstateModel()
model.load_state_dict(state_dict)
model.eval()
print("✅ Model architecture reconstructed and weights loaded")

# === Inspect layers ===
print("\n🔍 Layer weight statistics:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name:30s} | shape: {tuple(param.shape)} | mean: {param.data.mean():.4f} | std: {param.data.std():.4f}")

# === Run a forward pass on dummy input ===
dummy_input = torch.randn(1, 8)
with torch.no_grad():
    output = model(dummy_input)
    print(f"\n🧪 Dummy input: {dummy_input}")
    print(f"🧮 Model output: {output.item():.4f}")