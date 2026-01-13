# JoyCaption Worker - Built on vLLM 0.10.2 official image
FROM vllm/vllm-openai:v0.10.2

# Add RunPod
RUN pip install runpod requests

# Model downloads to network volume on cold start
ENV MODEL_NAME=fancyfeast/llama-joycaption-beta-one-hf-llava \
    HF_HOME=/runpod-volume/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
