import requests, torch, cloudpickle, io
from flask import Flask, request, jsonify
import threading
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import os

# === Load model and training function dynamically ===
namespace = {}
with open("model_def.py", "r") as f:
    exec(f.read(), namespace)

TinyModel = namespace["TinyModel"]
train_model = namespace["train_model"]

# === Load real dataset ===
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target.reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = (X_tensor, y_tensor)

# === Coordinator endpoint ===
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
    print(" Final model received from coordinator")
    final_model = state_dict
    return jsonify({"status": "received"})


def run_flask():
    app.run(host="0.0.0.0", port=CLIENT_CALLBACK_PORT)

threading.Thread(target=run_flask, daemon=True).start()

# === Send job ===
print(" Sending model, data, and train_func to coordinator...")

files = {
    'model': io.BytesIO(cloudpickle.dumps(TinyModel())),
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
    timeout=15
)

print(" Coordinator response:", response.json())

# === Wait for model ===
import time
print(" Waiting for final model... (Ctrl+C to quit)")
while final_model is None:
    time.sleep(1)

# === Evaluation ===
print(" Evaluating model on original dataset...")
model = TinyModel()
model.load_state_dict(final_model)
model.eval()

with torch.no_grad():
    y_pred = model(X_tensor).squeeze().numpy()
    y_true = y_tensor.squeeze().numpy()

    if len(set(y_true)) > 2:
        print(" Labels are not binary. Skipping AUC.")
    else:
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC score: {auc:.4f}")

# === Save model ===
os.makedirs("result", exist_ok=True)
torch.save(final_model, "result/final_model.pth")
print(" Final model saved to result/final_model.pth")