# Flower State Classification

This Document contains a Readme and short overview of the Flower State Classification Project 

## Hardware Setup:
To Test and implement the system, the following Hardware was used:
Jetson Nano devkit 4GB.
Jetpack 4.6.1

Most of the neural network training was done on a different machine:
Windows 10
NVIDIA RTX 3060Ti
AMD Ryzen 5 3600x

## Installation:
The login password for the Jetson Nano devkit that was used to implement the system is *01527619*

(pat. taken from https://elinux.org/Jetson_Zoo)

Install some dependencies using:

```bash
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```

### Python Environment Installation (Ubuntu)
Pipenv is not needed for installation on the Jetson Nano, but can be used to install the system on a different machine.

The following commands are used to install the python distribution and the needed dependencies on the Jetson Nano:

```bash

sudo apt update
sudo apt install -y python3.8 python3.8-venv python3.8-dev python3-pip libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev libcublas-dev

python3.8 -m venv venv
source venv/bin/activate
python3.8 -m pip install -U pip wheel 

python3.8 -m pip install wheels/torch-*.whl wheels/torchvision-*.whl

python3.8 -m pip install -e .

python3.8 -m pip install ultralytics websockets
```

## Usage
After installing the system, the scripts in the root folder of the module (src/flower_state_classification) can be used to run the system. Use the installed virtual environment to run the scripts.

- run.py -> Start the system using the default configuration and with provided commandline arguments
- run_forever.py -> Starts the run.py script with the provided arguments and restarts it if it crashes
- run_test_videos.py -> Start the system on multiple test videos with different settings
- run_server.py -> Runs a Websocket Server to read the results of the system

When debug output is enabled for the system, they will be put in the output folder, together with a logfile.

The src/flower_state_classification folder contains a package with all the python code used for running the system and training the neural networks.
For further information on the training, there is an additional README file in the src/flower_state_classification/cv/training folder

To automatically start the project when restating the Jetson Nano, a script has to be created that runs the run_forever.py script, with the correct python environment activated. 

### Configuration
When running the system (run.py) without any commandline arguments, the system will try to run using a Webcam as input. When providing paths to an image folder or video file, they will be used instead. The system will then try to load the configuration from the settings.py file under src/flower_state_classification/settings/settings.

The behaviour of the system can be changed by editing this settings.py file.

The different parameters are explained in the following file itself.

## Troubleshooting
To find potential problems in the program, enable the different debug outputs in the settings.py file
- Plants are detected, but the optical flow is not calculated: Check if the plants are filtered out by the green mask. Use an image from the camera and test it with the util/hsv_setting_tester.py script.
- Optical flow is calculated, but the watering status is not detected: Change the thresholds for angle and magnitude in the settings.py file
- Jetson Nano appear to freeze when running the system. Try to shut down every other process (like an IDE), as the 4GB of the Jetson Nano can be used up quickly. Also try to run the system without the debug output enabled.