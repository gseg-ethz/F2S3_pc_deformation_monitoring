# Development Guide

This document covers development and container workflows for F2S3.
For end-user package installation on Ubuntu, see [installation.md](installation.md).

## Scope

This guide explains:

- local Docker builds and runs
- Docker Compose usage
- GitHub Actions publishing to GHCR

## Relevant Files

- `Dockerfile`
- `docker-compose.yaml`
- `.github/workflows/docker-ghcr.yml`

## Prerequisites

- Docker Engine with Compose plugin
- NVIDIA drivers installed on host
- NVIDIA Container Toolkit configured for Docker GPU access

Verify GPU visibility on host:

```bash
nvidia-smi
```

## Local Docker Workflow

Build:

```bash
docker compose build
```

Run default command (`f2s3 --help`):

```bash
docker compose run --rm f2s3
```

Run F2S3 against mounted repository data:

```bash
docker compose run --rm f2s3 \
  --source_cloud /workspace/data/_sample_folder/raw_data/epoch1.ply \
  --target_cloud /workspace/data/_sample_folder/raw_data/epoch2.ply
```

Override build args when needed:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
F2S3_VERSION=0.1.0 \
docker compose build
```

## Dockerfile Behavior

The `Dockerfile` uses a multi-stage build:

1. `builder` stage (`nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`)
2. `runtime` stage (`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`)

Builder stage installs C++ build dependencies (`python3-dev`, `swig`, `cmake`, `libpcl-dev`) and creates wheels.
Runtime stage installs only runtime dependencies and then installs `f2s3` from prebuilt wheels.

## GitHub Actions GHCR Workflow

Workflow file: `.github/workflows/docker-ghcr.yml`

Triggers:

- push tags matching `v*.*.*`
- manual trigger (`workflow_dispatch`)

Behavior:

- Image target: `ghcr.io/<owner>/<repo>` (repository name lowercased)
- On tag builds (`vX.Y.Z`), `F2S3_VERSION` is set to `X.Y.Z`
- On non-tag builds, `F2S3_VERSION` is set to `0.0.0.dev<run_number>`

Generated image tags include Git tag, commit SHA, and `latest`.

## Publishing a Release Image

Create and push a semantic tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

This triggers the workflow and publishes a version-tagged image to GHCR.

## Troubleshooting

No `docker` command:

- install Docker and Compose plugin on the machine running the commands

No GPU inside container:

- verify host GPU works with `nvidia-smi`
- verify NVIDIA Container Toolkit is installed and configured
- test with:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 nvidia-smi
```
