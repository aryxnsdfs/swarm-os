import requests
prompt = "Deploy a standard feedforward neural network for our tabular dataset. Lock the VRAM exactly at 500m. The activations are spilling over the limit."
try:
    print("Testing manual orchestration...")
    r = requests.post("http://localhost:8000/api/orchestrate", json={"prompt": prompt}, timeout=180)
    import json
    print("SUCCESS:")
    print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Failed:", e)
