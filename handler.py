"""JoyCaption vLLM Handler - Simple RunPod Serverless Worker"""

import os
import runpod
import subprocess
import sys
import time
import requests

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = os.environ.get("MODEL_NAME", "fancyfeast/llama-joycaption-beta-one-hf-llava")
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "600"))  # 10 minutes default for cold start
VLLM_EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "")  # Override/add vLLM args

def start_vllm():
    """Start vLLM server in background"""
    print(f"Loading model: {MODEL_NAME}")
    print(f"Startup timeout: {STARTUP_TIMEOUT}s")

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    if VLLM_EXTRA_ARGS:
        cmd.extend(VLLM_EXTRA_ARGS.split())

    print(f"vLLM command: {' '.join(cmd)}")

    # Start vLLM with output visible for debugging
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1
    )

    for i in range(STARTUP_TIMEOUT):
        # Check if process crashed
        if process.poll() is not None:
            raise RuntimeError(f"vLLM process exited with code {process.returncode}")

        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.ok:
                print(f"vLLM server ready after {i+1}s!")
                return True
        except requests.exceptions.ConnectionError:
            # Server not up yet, this is expected during startup
            pass
        except requests.exceptions.Timeout:
            # Health check timed out, server may be busy loading
            print(f"Health check timeout at {i+1}s, server may be loading model...")
        except Exception as e:
            # Unexpected error, log it but continue waiting
            print(f"Health check error at {i+1}s: {type(e).__name__}: {e}")

        time.sleep(1)

    # Timeout reached - kill the process and report failure
    process.terminate()
    raise RuntimeError(f"vLLM server failed to start within {STARTUP_TIMEOUT}s")

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
