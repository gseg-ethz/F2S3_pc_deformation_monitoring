# Installation Guide for Ubuntu-based Systems

F2S3 has been successfully tested on Ubuntu 20.04 LTS, 22.04 LTS, and 24.04 LTS, both as native OS and via Windows Subsystem for Linux (WSL2). The installation process involves setting up a Python environment, installing PyTorch with CUDA support, and installing necessary system dependencies for the C++ extensions.

## Table of Contents

- [Installation Guide for Ubuntu-based Systems](#installation-guide-for-ubuntu-based-systems)
  - [Table of Contents](#table-of-contents)
  - [1. Installation Overview](#1-installation-overview)
  - [2. Clone the Repository](#2-clone-the-repository)
  - [3. Create a Python Environment](#3-create-a-python-environment)
  - [4. Install PyTorch](#4-install-pytorch)
  - [5. Install System Dependencies](#5-install-system-dependencies)
  - [6. Install F2S3](#6-install-f2s3)
  - [7. Verify GPU Support](#7-verify-gpu-support)
- [Troubleshooting](#troubleshooting)
  - [Check GPU Driver](#check-gpu-driver)
  - [Check CUDA Compiler](#check-cuda-compiler)
  - [CUDA Installation](#cuda-installation)
  - [Conda `libstdc++` Issue](#conda-libstdc-issue)

## 1. Installation Overview

We present a streamlined installation process for F2S3 using virtualenv. If you prefer Conda, the steps are similar but may require adjustments for environment management (see also [Troubleshooting → Conda `libstdc++` Issue](#conda-libstdc-issue)).

**Installation requirements**

* Linux (tested on Ubuntu 20.04 LTS, 22.04 LTS, and 24.04 LTS)
* NVIDIA GPU with CUDA support
* Python >=3.11,<3.13

If CUDA is not installed yet, see the [Troubleshooting → CUDA installation](#cuda-installation) section below.

---

## 2. Clone the Repository

```bash
git clone https://github.com/gseg-ethz/F2S3_pc_deformation_monitoring.git F2S3
cd F2S3
```

---

## 3. Create a Python Environment

Example using `venv`:

```bash
python3 -m venv f2s3_env
source f2s3_env/bin/activate
pip install --upgrade pip
```

Conda environments also work.

---

## 4. Install PyTorch

Follow the official PyTorch install tool:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example command (check for latest versions!):

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## 5. Install System Dependencies

These are required for the C++ extensions. Adjust if using Conda (e.g., `conda install -c conda-forge cmake`).

```bash
sudo apt install python3-dev swig cmake libpcl-dev
```

---

## 6. Install F2S3

```bash
pip install .
```

Verify the CLI:

```bash
f2s3 -h
```

---

## 7. Verify GPU Support

Check that PyTorch detects CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```
True
```

If CUDA is not detected, see [Troubleshooting](#troubleshooting).

---

# Troubleshooting

## Check GPU Driver

```bash
nvidia-smi
```

## Check CUDA Compiler

```bash
nvcc --version
```

---

## CUDA Installation

If CUDA is not installed, follow the official NVIDIA guide:

[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

---

## Conda `libstdc++` Issue

If using Anaconda and encountering `libstdc++` errors:

```bash
conda install -c conda-forge libstdcxx-ng
```
