from flask import Flask, request, jsonify
import torch, cloudpickle, io, requests, os
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
#from sklearn.metrics import mean_squared_error, roc_auc_score

app = Flask(__name__)

client_callback_url = None
received_states = {}
TOTAL_WORKERS = 3
WORKER_ENDPOINTS = [
    "http://worker1:5000",
    "http://worker2:5000",
    "http://worker3:5000",
]

ModelClass = None

@app.post("/start_training")
def start_training():
    global client_callback_url, received_states, ModelClass

    print("📥 Incoming training job...", flush=True)

    try:
        model_def_file = request.files['model_def']
        data_file = request.files['data']
        callback_url = request.form['callback_url']
        client_callback_url = callback_url
        print("✅ Received model_def and callback URL", flush=True)
    except Exception as e:
        print("❌ Failed to extract files/form:", e, flush=True)
        return jsonify({"error": f"Missing file or form field: {e}"}), 400

    # === Dynamically load model and training function ===
    try:
        namespace = {}
        exec(model_def_file.read(), namespace)
        ModelClass = namespace['RealEstateModel']
        train_func = namespace['train_model']
    except Exception as e:
        return jsonify({"error": f"Failed to load model_def.py: {e}"}), 400

    # === Load dataset ===
    try:
        df = pd.read_csv(data_file)

        X = df.drop(columns=["target"]).values
        y = df[["target"]].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = (X_tensor, y_tensor)

        model = ModelClass()
    except Exception as e:
        return jsonify({"error": f"Failed to prepare dataset: {e}"}), 500

    received_states = {}
    shards = list(zip(torch.chunk(dataset[0], TOTAL_WORKERS), torch.chunk(dataset[1], TOTAL_WORKERS)))
    print(f"📊 Sharded data into {TOTAL_WORKERS} parts", flush=True)

    def dispatch():
        def send_to_worker(i, url):
            files = {
                'model': io.BytesIO(cloudpickle.dumps(model)),
                'data': io.BytesIO(),
                'train_func': io.BytesIO(cloudpickle.dumps(train_func)),
            }
            torch.save(shards[i], files['data'])
            for f in files.values(): f.seek(0)

            print(f"🚚 Sending data to {url}...", flush=True)
            try:
                requests.post(f"{url}/send_data", files=files, timeout=20)
                requests.post(f"{url}/train", params={"worker_id": i}, timeout=300)
            except Exception as e:
                print(f"❌ Error with {url}: {e}", flush=True)

        with ThreadPoolExecutor(max_workers=TOTAL_WORKERS) as executor:
            for i, url in enumerate(WORKER_ENDPOINTS):
                executor.submit(send_to_worker, i, url)

    Thread(target=dispatch).start()
    return jsonify({"status": "Training dispatched"})


@app.post("/receive_model")
def receive_model():
    global received_states

    worker_id = request.args.get("worker_id")
    try:
        buffer = io.BytesIO(request.data)
        state_dict = torch.load(buffer)
        received_states[worker_id] = state_dict
        print(f"📥 Received model from {worker_id} ({len(received_states)}/{TOTAL_WORKERS})", flush=True)
    except Exception as e:
        print(f"❌ Error receiving model from {worker_id}: {e}", flush=True)
        return jsonify({"error": str(e)}), 400

    if len(received_states) == TOTAL_WORKERS:
        print("🧮 All models received. Starting merge...", flush=True)
        merge_and_return()

    return jsonify({"status": "received"})

def merge_and_return():
    print("🔗 Merging models...", flush=True)
    keys = list(received_states.keys())
    base = received_states[keys[0]]
    avg_state = {}

    for k in base:
        stacked = torch.stack([
            v[k].float() if torch.is_floating_point(v[k]) else v[k]
            for v in received_states.values()
        ])
        if torch.is_floating_point(base[k]):
            avg_state[k] = stacked.mean(0)
        else:
            avg_state[k] = base[k]

    if ModelClass is None:
        print("❌ ModelClass not available to reconstruct model", flush=True)
        return

    # Save locally
    print("🔍 Saving model to:", os.path.abspath("result/final_model.pth"), flush=True)
    os.makedirs("result", exist_ok=True)
    torch.save(avg_state, "result/final_model.pth")
    print("💾 Saved merged model to result/final_model.pth", flush=True)

    model = ModelClass()
    model.load_state_dict(avg_state)

    # Save if needed:
    torch.save(model.state_dict(), "result/final_model.pth")

    # Send
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    
    # buffer = io.BytesIO()
    # torch.save(avg_state, buffer)
    # buffer.seek(0)

    try:
        print("📤 Sending merged model to client...", flush=True)
        response = requests.post(
            client_callback_url,
            files={"model": buffer},
            timeout=10
        )
        print("✅ Final model sent to client", flush=True)
    except Exception as e:
        print(f"❌ Failed to send merged model: {e}", flush=True)

@app.get("/")
def health():
    return "Coordinator ready", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
