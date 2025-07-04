import requests
import torch
import cloudpickle
import io
import threading
import time
import os
from flask import Flask, request, jsonify
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# === Load model and training function dynamically ===
namespace = {}
with open("model_def.py", "r") as f:
    exec(f.read(), namespace)

ModelClass = namespace["RealEstateModel"]
train_model = namespace["train_model"]

# === Load and preprocess dataset ===
housing = fetch_california_housing()
X = housing.data
y = housing.target.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = (X_tensor, y_tensor)

# === Coordinator config ===
COORDINATOR_URL = "http://localhost:6000/start_training"
CLIENT_CALLBACK_PORT = 8080

# === Flask app to receive final model ===
app = Flask(__name__)
final_model = None

@app.post("/receive_final_model")
def receive_final_model():
    global final_model
    buffer = io.BytesIO(request.files['model'].read())
    state_dict = torch.load(buffer)
    print("🎉 Final model received from coordinator")
    final_model = state_dict
    return jsonify({"status": "received"})

def run_flask():
    app.run(host="0.0.0.0", port=CLIENT_CALLBACK_PORT)

# Start receiver server in background
threading.Thread(target=run_flask, daemon=True).start()

# === Send training job ===
print("🚀 Sending model, data, and train_func to coordinator...")

files = {
    'model': io.BytesIO(cloudpickle.dumps(ModelClass())),
    'data': io.BytesIO(),
    'train_func': io.BytesIO(cloudpickle.dumps(train_model)),
}
torch.save(dataset, files['data'])
for f in files.values():
    f.seek(0)

response = requests.post(
    COORDINATOR_URL,
    files=files,
    data={'callback_url': f"http://host.docker.internal:{CLIENT_CALLBACK_PORT}/receive_final_model"},
    timeout=30
)

print("✅ Coordinator response:", response.json())

# === Wait for model ===
print("⏳ Waiting for final model... (Ctrl+C to quit)")
while final_model is None:
    time.sleep(1)

# === Evaluation ===
print("📊 Evaluating model on original dataset...")
model = ModelClass()
model.load_state_dict(final_model)
model.eval()

with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()
    y_true = y_tensor.squeeze().numpy()
    mse = mean_squared_error(y_true, y_pred)
    print(f"📉 MSE: {mse:.4f}")



# Convert to binary classification based on median threshold
threshold = y_tensor.median().item()
y_binary = (y_tensor > threshold).float().numpy()

# For classification AUC, use the raw prediction scores
with torch.no_grad():
    y_pred_scores = model(X_tensor).squeeze().numpy()

try:
    auc = roc_auc_score(y_binary, y_pred_scores)
    print(f"📈 AUC (binary target, median threshold={threshold:.2f}): {auc:.4f}")
except Exception as e:
    print(f"⚠️ AUC calculation failed: {e}")
# === Save model ===
os.makedirs("result", exist_ok=True)
torch.save(final_model, "result/final_model.pth")
print("✅ Final model saved to result/final_model.pth")

