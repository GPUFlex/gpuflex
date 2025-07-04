from flask import Flask, request, jsonify
import torch, cloudpickle, io, requests
from threading import Thread
from concurrent.futures import ThreadPoolExecutor


app = Flask(__name__)

client_callback_url = None
received_states = {}
TOTAL_WORKERS = 3

WORKER_ENDPOINTS = [
    "http://worker1:5000",
    "http://worker2:5000",
    "http://worker3:5000",
]

@app.post("/start_training")
def start_training():
    global client_callback_url, received_states

    print("📥 Incoming training job...", flush=True)

    try:
        model_file = request.files['model']
        data_file = request.files['data']
        train_func_file = request.files['train_func']
        callback_url = request.form['callback_url']
        print("✅ Received files and callback URL", flush=True)
    except Exception as e:
        print("❌ Failed to extract files/form:", e, flush=True)
        return jsonify({"error": f"Missing file or form field: {e}"}), 400

    try:
        model = cloudpickle.loads(model_file.read())
        data = torch.load(data_file)
        train_func = cloudpickle.loads(train_func_file.read())
        client_callback_url = callback_url

        model_size = sum(p.numel() for p in model.parameters())
        print(f"✅ Deserialized model ({model_size} parameters), dataset ({len(data[0])} samples), and training function", flush=True)
        print(f"🔗 Callback URL: {client_callback_url}", flush=True)
    except Exception as e:
        print("❌ Failed to deserialize model/data/train_func:", e, flush=True)
        return jsonify({"error": f"Deserialization error: {e}"}), 400

    received_states = {}
    shards = list(zip(torch.chunk(data[0], TOTAL_WORKERS), torch.chunk(data[1], TOTAL_WORKERS)))
    print(f"📊 Sharded data into {TOTAL_WORKERS} parts: {[len(shard[0]) for shard in shards]}", flush=True)


    def dispatch():
        def send_to_worker(i, url):
            files = {
                'model': io.BytesIO(cloudpickle.dumps(model)),
                'data': io.BytesIO(),
                'train_func': io.BytesIO(cloudpickle.dumps(train_func)),
            }
            torch.save(shards[i], files['data'])
            for f in files.values():
                f.seek(0)

            print(f"📦 Payload to {url}: model={files['model'].getbuffer().nbytes} bytes, data={files['data'].getbuffer().nbytes} bytes", flush=True)

            try:
                print(f"🚚 Sending data to {url}...", flush=True)
                resp1 = requests.post(f"{url}/send_data", files=files, timeout=20)
                print(f"📤 Data sent to {url}: {resp1.status_code}", flush=True)

                print(f"🏋️ Triggering training on {url}...", flush=True)
                resp2 = requests.post(f"{url}/train", timeout=300)
                print(f"✅ Training triggered on {url}: {resp2.status_code}", flush=True)
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
    print(f"📨 Receiving model from worker {worker_id}...", flush=True)

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
    try:
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
                # для LongTensor або інших — беремо просто з першого воркера
                avg_state[k] = base[k]

        buffer = io.BytesIO()
        torch.save(avg_state, buffer)
        buffer.seek(0)

        print("📤 Sending merged model to client...", flush=True)
        response = requests.post(
            client_callback_url,
            files={"model": buffer},
            timeout=10
        )
        print("✅ Final model sent to client", flush=True)

    except Exception as e:
        print(f"❌ Failed to merge or send model to client: {e}", flush=True)

@app.get("/")
def health():
    return "Coordinator ready", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)