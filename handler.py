"""JoyCaption vLLM Handler - Simple RunPod Serverless Worker"""

import runpod
import subprocess
import time
import requests

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"

def start_vllm():
    """Start vLLM server in background"""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "bfloat16",
        "--max-model-len", "2048",
        "--max-num-seqs", "96",
        "--max-num-batched-tokens", "120000",
        "--gpu-memory-utilization", "0.92",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--trust-remote-code",
        "--limit-mm-per-prompt", '{"image": 1}',
    ]
    subprocess.Popen(cmd)

    for _ in range(120):
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.ok:
                print("vLLM server ready!")
                return True
        except:
            pass
        time.sleep(1)
    raise RuntimeError("vLLM server failed to start")

print("Starting vLLM server...")
start_vllm()

def handler(job):
    """Process caption request"""
    try:
        job_input = job["input"]

        if "openai_input" in job_input:
            payload = job_input["openai_input"]
        elif "messages" in job_input:
            payload = job_input
        else:
            image_b64 = job_input.get("image")
            if not image_b64:
                return {"error": "Missing 'image' in input"}
            prompt = job_input.get("prompt", "Write a detailed description of this image.")
            payload = {
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                "max_tokens": job_input.get("max_tokens", 250),
                "temperature": job_input.get("temperature", 0.7),
                "top_p": 0.9
            }

        if "model" not in payload:
            payload["model"] = MODEL_NAME

        response = requests.post(VLLM_URL, json=payload, timeout=90)
        if response.ok:
            return response.json()
        else:
            return {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
