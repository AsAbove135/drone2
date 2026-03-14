FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# System deps (devel image needed for mamba-ssm CUDA kernel compilation)
RUN apt-get update -qq && \
    apt-get install -y -qq git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project
COPY . /workspace/drone2/

# Install Python deps
RUN cd /workspace/drone2 && pip install -r requirements.txt -q

# Install mamba-ssm with CUDA kernels (compiles Triton/CUDA selective scan)
RUN pip install mamba-ssm causal-conv1d -q --no-build-isolation

# Default: run the worker
ENV WORK_DIR=/workspace/drone2
CMD ["python", "-m", "autoresearch.cloud.worker"]
