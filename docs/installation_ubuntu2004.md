# Tested installation procedure Ubuntu 20.04 LTS
The following procedure was tested on a fresh install of Ubuntu 20.04 LTS in March 2022.

## CUDA
F2S3 requires a CUDA-enabled GPU. We would recommend following the [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for the CUDA Toolkit provided by NVIDIA. In the following, we will provide the necessary steps valid at the time of testing.

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

Additionally, environment variables need to be updated:
```shell
export PATH=/usr/local/cuda-11.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

These commands need to be run everytime a new terminal session is opened. To make the changes persistent for the current user, the above lines can be added to `~/.bashrc` (current terminal needs to be restarted after modification). The successful install can be checked with `nvcc --version`.

## GIT Repo

```shell
# Install GIT (if necesseray)
sudo apt install git

# Clone the github repo
cd /path/to/project_parent_folder
# Select one of the following lines depending on your mode of github access (if you don't know the difference then select HTTPS)
# SSH
git clone git@github.com:zgojcic/F2S3_pc_deformaton_monitoring.git ./F2S3
# HTTPS
git clone https://github.com/zgojcic/F2S3_pc_deformaton_monitoring.git ./F2S3

cd F2S3
git checkout development
```

## Virtual environment
We tested both anaconda and pip based installations. In the following we present the steps based on pip.

```shell
sudo apt install python3.8-venv
cd /path/to/project_parent_folder/F2S3
python3 -m venv ./venv

# Activate virtual environment
source ./venv/bin/activate
```

## PyTorch
We would recommend checking the [PyTorch website](https://pytorch.org/get-started/locally/) for the current installation recommendations. F2S3 was both tested with CUDA version 10.X and 11.X. Check which version is installed by running `nvcc --version`.  

```shell
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

We would recommend checking if cuda and pytorch were installed successfully at this point:
```shell
python
import torch
torch.cuda.is_available()
exit()
```
The code snippet should return `true`. If not, check the environment variables with `echo "$PATH"` and `echo "LD_LIBRARY_PATH"`.

## Install additional requirements

In a first step addiontal apt packages need to be installed:
```shell
sudo apt install python3-dev
sudo apt install swig
sudo apt install cmake

sudo apt install libpcl-dev
```

```shell
pip install open3d
pip install -r requirements.txt
```

## C++ tools and python wrappers

C++ tools are located in `./cpp_core/' and their python wrappers can be compiled as follows:

### pc_tiling
We have had issues with `libboost_python3`. The current way to solve it is by adding symbolic links:
```shell
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libboost_python38.so libboost_python3.so
sudo ln -s libboost_python38.a libboost_python3.a
```

```shell
cd /path/to/project_parent_folder/F2S3/cpp_core/pc_tiling
cmake -DCMAKE_BUILD_TYPE=Release .
make -j8
swig -c++ -python pc_tiling.i
```

Test the successful python tie-in by running `python -c "import pc_tiling"`. If you don't get an error message it is successful.

After the generation pc_tiling can be imported into python with `import pc_tiling`. Specific information on how to use pc_tiling (i.e. available functions and their input parameters) are available in respective [readme](../cpp_core/pc_tiling/).


### supervoxel_segmentation
Before continuing to the next step, the location of the `numpy` header files needs to be identified:
```shell
python
import numpy
numpy.get_include()
exit()
```

Replace the include directory in the following code block: 
```shell
cd /path/to/project_parent_folder/F2S3/cpp_core/supervoxel_segmentation
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I\ /path/to/project_parent_folder/F2S3/venv/lib/python3.8/site-packages/numpy/core/include .
make -j8
swig -c++ -python supervoxel.i
```

Again, test the successful python tie-in by running `python -c "import supervoxel"`.

After the generation supervoxel_segmentation can be imported into python with `import supervoxel`. Specific information on how to use supervoxel_segmentation (i.e. available functions and their input parameters) are available in respective [readme](../cpp_core/supervoxel_segmentation/).

