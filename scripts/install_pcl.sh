# Clone latest PCL
apt-get update
apt-get install git

cd ~/Documents
# for clone pcl-1.8.1
git clone --branch pcl-1.8.1 https://github.com/PointCloudLibrary/pcl.git pcl-trunk 
# git clone https://github.com/PointCloudLibrary/pcl.git pcl-trunk
ln -s pcl-trunk pcl
cd pcl

# Install prerequisites
apt-get install g++
apt-get install cmake cmake-gui
apt-get install doxygen
apt-get install mpi-default-dev openmpi-bin openmpi-common
apt-get install libflann1.8 libflann-dev
apt-get install libeigen3-dev
apt-get install libboost-all-dev
apt-get install libvtk6-dev libvtk6.2 libvtk6.2-qt
#sudo apt-get install libvtk5.10-qt4 libvtk5.10 libvtk5-dev  # I'm not sure if this is necessary.
apt-get install 'libqhull*'
apt-get install libusb-dev
apt-get install libgtest-dev
apt-get install git-core freeglut3-dev pkg-config
apt-get install build-essential libxmu-dev libxi-dev
apt-get install libusb-1.0-0-dev graphviz mono-complete
apt-get install qt-sdk openjdk-9-jdk openjdk-9-jre
apt-get install phonon-backend-gstreamer
apt-get install phonon-backend-vlc
apt-get install libopenni-dev libopenni2-dev

# Compile and install PCL
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=None -DBUILD_GPU=OFF -DBUILD_apps=ON -DBUILD_examples=ON ..
make -j 8
make install
