# GPU-optimized vLLM serving image for the fine-tuned tool-calling / SQL model.
FROM vllm/vllm-openai:latest

WORKDIR /workspace

# Model weights are bind-mounted at runtime via docker-compose (see /models).
# Install curl for the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV MODEL_PATH=/models/merged_16bit \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.90 \
    TENSOR_PARALLEL_SIZE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "/models/merged_16bit", \
     "--served-model-name", "tool-sql-llama3.1-8b", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "4096", \
     "--gpu-memory-utilization", "0.90", \
     "--enable-prefix-caching", \
     "--tensor-parallel-size", "1"]
