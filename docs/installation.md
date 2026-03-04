# Installation Guide for Ubuntu-based Systems

F2S3 has been successfully tested on Ubuntu 20.04 LTS, 22.04 LTS, and 24.04 LTS, both as a native OS and via Windows Subsystem for Linux (WSL2).

The installation consists of:

1. Creating a Python environment
2. Verifying GPU driver availability
3. Installing PyTorch with CUDA support
4. Installing system dependencies required for the C++ extensions
5. Installing F2S3

---

# Quick Install (Ubuntu)

For experienced users, the following commands perform a minimal installation.

```bash
# clone repository
git clone https://github.com/gseg-ethz/F2S3_pc_deformation_monitoring.git F2S3
cd F2S3

# create Python environment
python3 -m venv f2s3_env
source f2s3_env/bin/activate
pip install --upgrade pip

# verify NVIDIA driver is working
nvidia-smi

# install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# install system dependencies required for C++ extensions
sudo apt install python3-dev swig cmake libpcl-dev

# install F2S3
pip install .

# verify installation
f2s3 -h
```

If GPU support is not detected later, see the **Troubleshooting** section below.

---

# Detailed Installation

## Table of Contents

- [Installation Guide for Ubuntu-based Systems](#installation-guide-for-ubuntu-based-systems)
- [Quick Install (Ubuntu)](#quick-install-ubuntu)
- [Detailed Installation](#detailed-installation)
  - [Table of Contents](#table-of-contents)
  - [1. Installation Overview](#1-installation-overview)
  - [2. Clone the Repository](#2-clone-the-repository)
  - [3. Create a Python Environment](#3-create-a-python-environment)
  - [4. Check NVIDIA Driver](#4-check-nvidia-driver)
  - [5. Install PyTorch](#5-install-pytorch)
  - [6. Install System Dependencies](#6-install-system-dependencies)
  - [7. Install F2S3](#7-install-f2s3)
  - [8. Verify GPU Support](#8-verify-gpu-support)
- [Troubleshooting](#troubleshooting)
  - [Check NVIDIA Driver](#check-nvidia-driver)
  - [Check PyTorch CUDA Support](#check-pytorch-cuda-support)
  - [Conda `libstdc++` Issue](#conda-libstdc-issue)
  - [C++ Extension Build Errors](#c-extension-build-errors)

---

## 1. Installation Overview

**Requirements**

- Linux (tested on Ubuntu 20.04 LTS, 22.04 LTS, and 24.04 LTS)
- NVIDIA GPU with a working driver
- Python 3.11-3.12

A system-wide CUDA toolkit installation is **not required** when using the official PyTorch CUDA wheels.

The only requirement is that the **NVIDIA driver is installed and functioning**.

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

Conda environments can also be used.

---

## 4. Check NVIDIA Driver

Before installing PyTorch, verify that the NVIDIA driver is installed, the GPU is detected, and check the CUDA runtime supported by the currently installed driver:

```bash
nvidia-smi
```

Example output:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54       Driver Version: 550.54       CUDA Version: 12.4     |
+-----------------------------------------------------------------------------+
```

If this command fails, the NVIDIA driver is not installed or not configured correctly.

Install the driver using the official NVIDIA guide:

https://www.nvidia.com/Download/index.aspx

> The CUDA version shown by `nvidia-smi` indicates the **maximum CUDA runtime supported by the installed driver**.  
> It does **not** indicate whether a CUDA toolkit is installed or which CUDA version PyTorch will use.

---

## 5. Install PyTorch

Install PyTorch with CUDA support using the official PyTorch wheels.

Check the latest recommended installation command at:

https://pytorch.org/get-started/locally/

> If the CUDA version shown by `nvidia-smi` is **lower than the CUDA version required by the PyTorch installation command**, you should either:
>
> - update your NVIDIA driver (see [Check NVIDIA Driver](#4-check-nvidia-driver)), or  
> - install an older PyTorch build that targets an earlier CUDA runtime.

Note that a **system-wide CUDA toolkit installation is not required** when using the official PyTorch wheels.

Example installation command:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 6. Install System Dependencies

F2S3 includes C++ extensions that require several system packages:

```bash
sudo apt install python3-dev swig cmake libpcl-dev
```

If you do not have `sudo` access, these packages may need to be installed using an alternative package manager that supports user-level installations, such as Conda (using `conda-forge`).

---

## 7. Install F2S3

Install the package from the repository root:

```bash
pip install .
```

Verify that the command line interface is available:

```bash
f2s3 -h
```

---

## 8. Verify GPU Support

Check that PyTorch detects CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```
True
```

You can also inspect the CUDA runtime used by PyTorch:

```bash
python -c "import torch; print(torch.version.cuda)"
```

---

# Troubleshooting

This section helps diagnose common issues related to GPU support and environment setup.

---

## Check NVIDIA Driver

Verify that the GPU driver is installed and working:

```bash
nvidia-smi
```

If this command fails, install the NVIDIA driver:

https://www.nvidia.com/Download/index.aspx

---

## Check PyTorch CUDA Support

Verify that PyTorch detects the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True` 

If `False` is returned while `nvidia-smi` works, reinstall PyTorch using the CUDA-enabled wheels from:

https://pytorch.org/get-started/locally/

---

## Conda `libstdc++` Issue

When using Anaconda or Miniconda, you may encounter errors related to incompatible `libstdc++` versions.

Example error:

```
GLIBCXX_x.x.x not found
```

This can usually be resolved by installing the updated runtime from `conda-forge`:

```bash
conda install -c conda-forge libstdcxx-ng
```

After installing, restart the environment:

```bash
conda deactivate
conda activate <your_env>
```

---

## C++ Extension Build Errors

If the installation fails while compiling the C++ extensions, verify that the required system dependencies are installed:

```bash
sudo apt install python3-dev swig cmake libpcl-dev
```

If you do not have `sudo` access, consider installing the dependencies using Conda (`conda-forge`) or Spack.