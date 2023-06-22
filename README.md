# Flower State Classification

This Document contains a Readme and short overview of the Flower State Classification Project 

## Project Overview
Three Different Stages
- Installation, Setup etc.
- Plant Object Detection
- Flower State Estimation

## Hardware Setup:
To Test and implement the system, the following Hardware was used:
Jetson Nano devkit 4GB.
Jetpack 4.6.1

Most of the neural network training was done on a different machine:
Windows 10
NVIDIA RTX 3060Ti
AMD Ryzen 5 3600x

## Installation:
To install and run the system on a Jetson Nano devkit, refer to the following instructions:

The login password for the Jetson Nano devkit that was used to implement the system is *01527619*

(pat. taken from https://elinux.org/Jetson_Zoo)

sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip 

sudo pip3 install -U pip testresources setuptools==49.6.0

sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow



# ln -s /usr/lib/python3.6/dist-packages/cv2 cv2


Installation with pipenv (TODO)

pip install 'pipenv==2021.5.29' see https://github.com/pypa/pipenv/issues/4829

sudo ln -s /usr/include/locale.h /usr/include/xlocale.h

# 3.8
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev

python3.8 -m venv venv_38
source venv_38/bin/activate
pip install -U pip wheel 

pip install -U pip gdown
gdown https://drive.google.com/uc?id=1hs9HM0XJ2LPFghcn7ZMOs5qu5HexPXwM
# torchvision 0.12.0
gdown https://drive.google.com/uc?id=1m0d8ruUY8RvCP9eVjZw4Nc8LAwM8yuGV

python3.8 -m pip install torch-*.whl torchvision-*.whl

pip install -e .

pip install ultralytics

## Code Overview
Python -> as a module
Run scripts in outer layer

CV->Training contains examples to train neural networks for different Frameworks -> Pytorch, Ultralytics Yolo, PaddleDetection

Different Training code -> Same dataset (fiftyone)(Can be exported to different formats, depending on network type/framework)