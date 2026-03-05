# syntax=docker/dockerfile:1.7

ARG CUDA_DEVEL_TAG=12.6.3-cudnn-devel-ubuntu24.04
ARG CUDA_RUNTIME_TAG=12.6.3-cudnn-runtime-ubuntu24.04

FROM nvidia/cuda:${CUDA_DEVEL_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG F2S3_VERSION=0.0.0

# Matches system dependencies documented in docs/installation.md.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libpcl-dev \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml setup.py README.md LICENSE ./
COPY src ./src

# Keep builds deterministic even when .git metadata is unavailable in Docker context.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${F2S3_VERSION}

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel build \
    && /opt/venv/bin/python -m pip wheel --wheel-dir /opt/wheels \
    --index-url ${TORCH_INDEX_URL} \
    --extra-index-url https://pypi.org/simple \
    .

FROM nvidia/cuda:${CUDA_RUNTIME_TAG} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep runtime deps aligned with installation docs and binary requirements.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libpcl-dev \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /opt/wheels /opt/wheels

# Install directly from wheel files to avoid VCS URL resolution at runtime.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-index --no-deps /opt/wheels/*.whl \
    && rm -rf /opt/wheels

ENV PATH="/opt/venv/bin:${PATH}"

ENTRYPOINT ["f2s3"]
CMD ["--help"]
