from flask import Flask, request, jsonify
import cloudpickle, torch, io, requests, sys, socket

app = Flask(__name__)

model = None
data = None
train_func = None
evaluate_func = None
#todo change url to deployed coordinator
COORDINATOR_URL = "http://coordinator:6000"

@app.post("/send_data")
def receive_data():
    global model, data, train_func, evaluate_func
    try:
        print(" Receiving data on worker...", flush=True)

        if 'model' in request.files:
            raw = request.files['model'].read()
            print(f" Model received: {len(raw)} bytes", flush=True)
            model = cloudpickle.loads(raw)
            print(" Model deserialized", flush=True)

        if 'data' in request.files:
            data_buf = request.files['data'].read()
            print(f" Dataset received: {len(data_buf)} bytes", flush=True)
            data = torch.load(io.BytesIO(data_buf))
            print(f" Data loaded: {len(data[0])} samples, each with shape {tuple(data[0][0].shape)}", flush=True)

        if 'train_func' in request.files:
            raw = request.files['train_func'].read()
            print(f" Train func received: {len(raw)} bytes", flush=True)
            train_func = cloudpickle.loads(raw)
            print(" Training function loaded", flush=True)

        if 'evaluate_func' in request.files:
            raw = request.files['evaluate_func'].read()
            print(f" Evaluate func received: {len(raw)} bytes", flush=True)
            evaluate_func = cloudpickle.loads(raw)
            print(" Evaluation function loaded", flush=True)

        print(" All components received and initialized", flush=True)
        return jsonify({"status": "Data received"})

    except Exception as e:
        print(f" Failed to load data: {e}", flush=True)
        sys.exit(1)


@app.post("/train")
def train():
    global model, data, train_func, evaluate_func

    if None in (model, data, train_func):
        print(" Model/data/train_func not initialized", flush=True)
        sys.exit(1)

    try:
        print("🏋️ Starting training...", flush=True)
        train_func(model, data)
        print(" Training complete", flush=True)

        if evaluate_func:
            auc = evaluate_func(model, data)
            print(f" AUC on local shard: {auc:.4f}", flush=True)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model_size = buffer.getbuffer().nbytes
        print(f" Serialized model: {model_size} bytes", flush=True)

        worker_id = socket.gethostname()
        print(f" Sending trained model from worker {worker_id} to coordinator...", flush=True)

        resp = requests.post(
            f"{COORDINATOR_URL}/receive_model?worker_id={worker_id}",
            data=buffer.read(),
            timeout=10
        )

        print(f" Sent model to coordinator: {resp.status_code}", flush=True)
        return jsonify({"status": "trained"})
    except Exception as e:
        print(f" Training failed: {e}", flush=True)
        sys.exit(1)


@app.get("/")
def health():
    return "Worker ready", 200


if __name__ == "__main__":
    print(" Worker starting...", flush=True)
    app.run(host="0.0.0.0", port=5000)