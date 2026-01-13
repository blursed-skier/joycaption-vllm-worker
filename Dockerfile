# RunPod vLLM Worker - JoyCaption with vLLM 0.10.2 (model downloads on cold start)
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Install vLLM 0.10.2
RUN python3 -m pip install vllm==0.10.2

# Model config - downloads to network volume on cold start
ARG MODEL_NAME="fancyfeast/llama-joycaption-beta-one-hf-llava"
ARG BASE_PATH="/runpod-volume"

ENV MODEL_NAME=$MODEL_NAME \
    BASE_PATH=$BASE_PATH \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRUST_REMOTE_CODE=true

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src

# Model downloads on first cold start to network volume (not baked in)

# Start the handler
CMD ["python3", "/src/handler.py"]
