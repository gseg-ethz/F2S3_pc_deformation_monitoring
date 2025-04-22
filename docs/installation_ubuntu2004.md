# Tested installation procedure Ubuntu 20.04 LTS
The following procedure was tested on a fresh install of Ubuntu 20.04 LTS in March 2022.

## CUDA
F2S3 requires a CUDA-enabled GPU. We would recommend following the [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for the CUDA 
Toolkit provided by NVIDIA. In the following, we will provide the necessary steps valid at the time of testing.

```shell
# Install developement tools
sudo apt install build-essential

# Install correct kernel headers
sudo apt-get install linux-headers-$(uname -r)

# Download NVIDIA CUDA toolkit installer
wget https://developer.download.nvidia.com/compute/cuda/11.6.1/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb

# Install repository meta-data
sudo dpkg -i ./cuda-repo-ubuntu2004-11-6-local_11.6.1-510.47.03-1_amd64.deb

# Install the CUDA public GPG key
sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub

# Pin file to prioritize CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Update the Apt repository cache
sudo apt-get update

#Install CUDA
sudo apt-get install cuda
```

Additionally, environment variables need to be updated (the paths need to be adapted based on the CUDA version!):
```shell
export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

These commands need to be run everytime a new terminal session is opened. To make the changes persistent for the current 
user, the above lines can be added to `~/.bashrc` (current terminal needs to be restarted after modification). The 
successful install can be checked with `nvcc --version`.

## GIT Repo

```shell
# Install GIT (if necesseray)
sudo apt install git

# Clone the github repo
cd /path/to/project_parent_folder
# Select one of the following lines depending on your mode of github access (if you don't know the difference then select HTTPS)
# SSH
git clone git@github.com:gseg-ethz/F2S3_pc_deformation_monitoring.git ./F2S3
# HTTPS
git clone https://github.com/gseg-ethz/F2S3_pc_deformation_monitoring.git ./F2S3

cd F2S3
```

## Virtual environment
We tested f2s3 with (mini)anaconda and virtualenv. In the following we present the steps based on conda and pip.


## PyTorch
We would recommend checking the [PyTorch website](https://pytorch.org/get-started/locally/) for the current installation recommendations. F2S3 was both 
tested with CUDA version 10.X, 11.X, and 12.X. Check which version is installed by running `nvcc --version`.  

```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

We would recommend checking if cuda and pytorch were installed successfully at this point:
```shell
python -c "import torch; print(torch.cuda.is_available())"
```

The code snippet should return `true`. If not, check the environment variables with `echo "$PATH"` and `echo "LD_LIBRARY_PATH"`.

## Install additional requirements

In a first step additional apt packages need to be installed:
```shell
sudo apt install python3-dev
sudo apt install swig
sudo apt install cmake

sudo apt install libpcl-dev
```
Now install F2S3 by running:

```shell
pip install .
```

Test if the library works with: 
```shell
f2s3 -h
```

> Note: We have observed issues with  `libstdc++` under anaconda installs. We were able to solve the problem with a 
> conda installation: 
> 
> `conda install -c conda-forge libstdcxx-ng`


