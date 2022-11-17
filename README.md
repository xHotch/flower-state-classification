Hardware Setup:
Tested on Jetson Nano devkit 4GB.
Jetpack 4.6.1

Installation:

(pat. taken from https://elinux.org/Jetson_Zoo)

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip 

sudo pip3 install -U pip testresources setuptools==49.6.0

sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow



Installation with pipenv (TODO) :

pip install 'pipenv==2021.5.29' see https://github.com/pypa/pipenv/issues/4829
