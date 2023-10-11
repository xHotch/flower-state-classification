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

# ln -s /usr/lib/python3.6/dist-packages/cv2 cv2


### Python Environment Installation (Ubuntu)
sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev

python3.8 -m venv venv
source venv/bin/activate
python3.8 -m pip install -U pip wheel 

python3.8 -m pip install wheels/torch-*.whl wheels/torchvision-*.whl

python3.8 -m pip install -e .

python3.8 -m pip install ultralytics websockets

## Usage
After installing the system, the scripts in the root folder of the module (src/flower_state_classification) can be used to run the system:

- run.py -> Start the system using the default configuration and with provided commandline arguments
- run_forever.py -> Starts the run.py script with the provided arguments and restarts it if it crashes
- run_test_videos.py -> Start the system on multiple test videos with different settings
- run_server.py -> Runs a Websocket Server to read the results of the system

### Configuration
When running the system (run.py) without any commandline arguments, the system will try to run using a Webcam as input. When providing paths to an image folder or video file, they will be used instead. The system will then try to load the configuration from the settings.py file under src/flower_state_classification/settings/settings.

The behaviour of the system can be changed by editing this settings.py file.

The different parameters are explained in the following file itself.
