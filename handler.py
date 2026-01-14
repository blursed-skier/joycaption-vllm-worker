"""JoyCaption vLLM Handler - Enhanced RunPod Serverless Worker

Optimized for A100 80GB PCIe with comprehensive configuration support.
All parameters can be configured via RunPod environment variables.
"""

import os
import runpod
import subprocess
import sys
import time
import requests
import logging
import json
from typing import Dict, Any, Optional

# Configure logging for serverless environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Configuration - all parameters can be overridden via RunPod env vars
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = os.environ.get("MODEL_NAME", "fancyfeast/llama-joycaption-beta-one-hf-llava")
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "900"))  # 15 minutes for A100 cold start

# vLLM Performance Settings (A100 80GB optimized defaults)
VLLM_DTYPE = os.environ.get("VLLM_DTYPE", "bfloat16")
VLLM_MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "2048")
VLLM_MAX_NUM_SEQS = os.environ.get("VLLM_MAX_NUM_SEQS", "128")  # Match RunPod concurrency
VLLM_MAX_BATCHED_TOKENS = os.environ.get("VLLM_MAX_BATCHED_TOKENS", "131072")  # For 80GB
VLLM_GPU_UTIL = os.environ.get("VLLM_GPU_UTIL", "0.95")  # Leave 5% for system
VLLM_ENABLE_PREFIX_CACHING = os.environ.get("VLLM_ENABLE_PREFIX_CACHING", "1") == "1"
VLLM_ENABLE_CHUNKED_PREFILL = os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "1") == "1"
VLLM_TRUST_REMOTE_CODE = os.environ.get("VLLM_TRUST_REMOTE_CODE", "1") == "1"
VLLM_LIMIT_MM_PER_PROMPT = os.environ.get("VLLM_LIMIT_MM_PER_PROMPT", '{"image": 1}')
VLLM_EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "")

# Additional vLLM parameters for fine-tuning
VLLM_SWAP_SPACE = os.environ.get("VLLM_SWAP_SPACE", "0")  # Disable swap for A100 80GB
VLLM_ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"
VLLM_DISABLE_LOG_STATS = os.environ.get("VLLM_DISABLE_LOG_STATS", "1") == "1"

# Serverless-specific settings
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "128"))
HEALTH_CHECK_INTERVAL = int(os.environ.get("HEALTH_CHECK_INTERVAL", "5"))

# OpenAI API compatibility
OPENAI_SERVED_MODEL_NAME = os.environ.get("OPENAI_SERVED_MODEL_NAME_OVERRIDE", "llama-joycaption-beta-one")

# A100-specific optimizations
A100_OPTIMIZED = os.environ.get("A100_OPTIMIZED", "1") == "1"

# Rate limiting and timeout settings
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Debug mode
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

def start_vllm():
    """Start vLLM server in background with A100 optimizations using modern serve command"""
    logger.info(f"🚀 Starting JoyCaption vLLM Worker")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"GPU: A100 80GB PCIe (optimized: {A100_OPTIMIZED})")
    logger.info(f"Startup timeout: {STARTUP_TIMEOUT}s")
    logger.info(f"Max concurrency: {MAX_CONCURRENCY}")

    cmd = [
        "vllm", "serve", MODEL_NAME,
        
        "--host", "0.0.0.0",
        "--port", "8000",
    ]

    # A100-specific optimizations
    if A100_OPTIMIZED:
        logger.info("🎯 Applying A100 80GB optimizations...")
        cmd.extend(["--tensor-parallel-size", "1"])  # Single GPU for A100

    # Core vLLM parameters (modern serve syntax)
    if VLLM_DTYPE:
        cmd.extend(["--dtype", VLLM_DTYPE])
        logger.info(f"  dtype: {VLLM_DTYPE}")
    if VLLM_MAX_MODEL_LEN:
        cmd.extend(["--max-model-len", VLLM_MAX_MODEL_LEN])
        logger.info(f"  max-model-len: {VLLM_MAX_MODEL_LEN}")
    if VLLM_MAX_NUM_SEQS:
        cmd.extend(["--max-num-seqs", VLLM_MAX_NUM_SEQS])
        logger.info(f"  max-num-seqs: {VLLM_MAX_NUM_SEQS}")
    if VLLM_MAX_BATCHED_TOKENS:
        cmd.extend(["--max-num-batched-tokens", VLLM_MAX_BATCHED_TOKENS])
        logger.info(f"  max-num-batched-tokens: {VLLM_MAX_BATCHED_TOKENS}")
    if VLLM_GPU_UTIL:
        cmd.extend(["--gpu-memory-utilization", VLLM_GPU_UTIL])
        logger.info(f"  gpu-memory-utilization: {VLLM_GPU_UTIL}")

    # Boolean flags (modern serve syntax)
    if VLLM_ENABLE_PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")
        logger.info("  prefix-caching: enabled")
    if VLLM_ENABLE_CHUNKED_PREFILL:
        cmd.append("--enable-chunked-prefill")
        logger.info("  chunked-prefill: enabled")
    if VLLM_TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
        logger.info("  trust-remote-code: enabled")

    # Advanced settings (modern serve syntax)
    if VLLM_SWAP_SPACE:
        cmd.extend(["--swap-space", VLLM_SWAP_SPACE])
        logger.info(f"  swap-space: {VLLM_SWAP_SPACE}")
    if VLLM_ENFORCE_EAGER:
        cmd.append("--enforce-eager")
        logger.info("  enforce-eager: enabled")
    if VLLM_DISABLE_LOG_STATS:
        cmd.append("--disable-log-stats")
        logger.info("  log-stats: disabled")

    # Multi-modal settings (modern serve syntax)
    if VLLM_LIMIT_MM_PER_PROMPT:
        cmd.extend(["--limit-mm-per-prompt", VLLM_LIMIT_MM_PER_PROMPT])
        logger.info(f"  limit-mm-per-prompt: {VLLM_LIMIT_MM_PER_PROMPT}")

    # OpenAI API compatibility (modern serve syntax)
    cmd.extend(["--served-model-name", OPENAI_SERVED_MODEL_NAME])
    logger.info(f"  served-model-name: {OPENAI_SERVED_MODEL_NAME}")

    # Additional custom arguments
    if VLLM_EXTRA_ARGS:
        extra_args = VLLM_EXTRA_ARGS.split()
        cmd.extend(extra_args)
        logger.info(f"  extra-args: {extra_args}")

    logger.info(f"📋 vLLM command: {' '.join(cmd)}")

    # Start vLLM with proper output handling
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )

        # Monitor startup progress
        startup_logs = []
        for i in range(STARTUP_TIMEOUT):
            # Check if process crashed
            if process.poll() is not None:
                error_output = process.stderr.read() if process.stderr else ""
                raise RuntimeError(f"vLLM process exited with code {process.returncode}. Error: {error_output}")

            # Check health endpoint
            try:
                r = requests.get("http://localhost:8000/health", timeout=2)
                if r.ok:
                    logger.info(f"✅ vLLM server ready after {i+1}s!")
                    
                    # Read any remaining startup logs
                    if process.stdout:
                        remaining_output = process.stdout.read()
                        if remaining_output:
                            startup_logs.append(remaining_output)
                    
                    if DEBUG_MODE and startup_logs:
                        logger.debug("Startup logs:")
                        for log in startup_logs:
                            logger.debug(log.strip())
                    
                    return True
            except requests.exceptions.ConnectionError:
                # Server not up yet, normal during startup
                pass
            except requests.exceptions.Timeout:
                # Health check timeout, server may be loading model
                if i % 10 == 0:  # Log every 10 seconds to reduce noise
                    logger.info(f"⏳ Health check timeout at {i+1}s, model loading...")
            except Exception as e:
                logger.warning(f"⚠️  Health check error at {i+1}s: {type(e).__name__}: {e}")

            # Read and buffer startup logs
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    startup_logs.append(line)
                    if DEBUG_MODE and len(startup_logs) % 10 == 0:
                        logger.debug(f"Startup progress: {line.strip()}")

            time.sleep(1)

        # Timeout reached
        process.terminate()
        error_output = process.stderr.read() if process.stderr else ""
        raise RuntimeError(f"❌ vLLM server failed to start within {STARTUP_TIMEOUT}s. Last error: {error_output}")

    except Exception as e:
        logger.error(f"❌ Failed to start vLLM: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting JoyCaption vLLM Worker for RunPod Serverless")
        logger.info("=" * 60)
        
        # Start vLLM server
        start_vllm()
        
        logger.info("✅ vLLM server started successfully")
        logger.info("📡 Waiting for RunPod requests...")
        
    except Exception as e:
        logger.error(f"❌ Fatal error during startup: {e}")
        sys.exit(1)

def handler(job):
    """Process caption request with A100 optimizations and comprehensive error handling"""
    start_time = time.time()
    request_id = job.get("id", "unknown")
    
    try:
        job_input = job["input"]
        logger.debug(f"📥 Processing request {request_id}")

        # Input validation and routing
        if "openai_input" in job_input:
            payload = job_input["openai_input"]
            logger.debug(f"  Using openai_input format")
        elif "messages" in job_input:
            payload = job_input
            logger.debug(f"  Using messages format")
        elif "image" in job_input:
            # JoyCaption image captioning format (primary use case)
            image_b64 = job_input["image"]
            prompt = job_input.get("prompt", "Write a detailed description of this image.")
            
            # Validate image data
            if not image_b64 or len(image_b64) < 100:  # Minimum reasonable base64 image
                raise ValueError("Invalid image data - too short or empty")
            
            payload = {
                "model": OPENAI_SERVED_MODEL_NAME,  # Use served model name for compatibility
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                "max_tokens": job_input.get("max_tokens", 250),
                "temperature": job_input.get("temperature", 0.7),
                "top_p": job_input.get("top_p", 0.9),
                "stream": job_input.get("stream", False)
            }
            logger.debug(f"  Using JoyCaption image format (max_tokens={payload['max_tokens']})")
        elif "prompt" in job_input:
            # Simple text-only prompt format (for testing)
            payload = {
                "model": OPENAI_SERVED_MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": job_input["prompt"]
                }],
                "max_tokens": job_input.get("max_tokens", 250),
                "temperature": job_input.get("temperature", 0.7),
                "top_p": job_input.get("top_p", 0.9)
            }
            logger.debug(f"  Using text prompt format")
        else:
            error_msg = "Missing required input: 'openai_input', 'messages', 'image', or 'prompt'"
            logger.warning(f"❌ {error_msg}")
            return {"error": error_msg, "status": "invalid_input"}

        # Ensure model is set
        if "model" not in payload:
            payload["model"] = OPENAI_SERVED_MODEL_NAME

        # A100-specific request optimizations
        if A100_OPTIMIZED:
            # Set reasonable defaults for A100 performance
            payload.setdefault("max_tokens", 250)
            payload.setdefault("temperature", 0.7)
            payload.setdefault("top_p", 0.9)

        logger.debug(f"  Payload size: {len(str(payload))} chars")

        # Make request with retry logic
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    VLLM_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.ok:
                    result = response.json()
                    processing_time = time.time() - start_time
                    logger.debug(f"✅ Request {request_id} completed in {processing_time:.2f}s")
                    
                    # Add metadata for monitoring
                    result["_joycaption_meta"] = {
                        "processing_time": processing_time,
                        "request_id": request_id,
                        "attempt": attempt + 1
                    }
                    
                    return result
                else:
                    last_error = f"vLLM error {response.status_code}: {response.text}"
                    logger.warning(f"⚠️  Attempt {attempt + 1} failed: {last_error}")
                    
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        break
                    
                    # Exponential backoff for server errors
                    if attempt < MAX_RETRIES - 1:
                        sleep_time = min(2 ** attempt, 10)  # Cap at 10s
                        logger.info(f"😴 Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
            
            except requests.exceptions.Timeout:
                last_error = f"Request timeout after {REQUEST_TIMEOUT}s"
                logger.warning(f"⏳ {last_error} (attempt {attempt + 1})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
            
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                logger.warning(f"⚠️  {last_error} (attempt {attempt + 1})")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
            
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"💥 {last_error}")
                break

        # All attempts failed
        error_result = {
            "error": last_error or "Unknown error",
            "status": "request_failed",
            "attempts": MAX_RETRIES,
            "request_id": request_id
        }
        logger.error(f"❌ Request {request_id} failed after {MAX_RETRIES} attempts")
        return error_result

    except Exception as e:
        error_msg = f"Handler exception: {str(e)}"
        logger.error(f"💥 {error_msg}", exc_info=DEBUG_MODE)
        return {
            "error": error_msg,
            "status": "handler_error",
            "request_id": request_id
        }

# Start RunPod serverless with enhanced configuration
try:
    logger.info("🎯 Starting RunPod serverless handler...")
    logger.info(f"  Max concurrency: {MAX_CONCURRENCY}")
    logger.info(f"  Request timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"  Max retries: {MAX_RETRIES}")
    
    runpod.serverless.start({
        "handler": handler,
        "concurrency": MAX_CONCURRENCY  # Set concurrency limit
    })
    
except Exception as e:
    logger.error(f"❌ RunPod serverless startup failed: {e}")
    sys.exit(1)
