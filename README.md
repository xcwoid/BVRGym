# BVRGym
## Description
This library is heavily based on JSBSim software (https://github.com/JSBSim-Team/jsbsim). 
The primary purpose of this library is to give users the possibility to explore Beyond Visual Range (BVR) tactics using Reinforcement learning.

## Environment
Currently there are three available environments:
Evading one missile 
Evading two missile 
BVR air combat
The environments above mainly use the F16 flight dynamics model and a BVR missile model. 
The F16 model has an additional wrapper to simply control, while the BVR missile has a Proportional Navigation guidance law implemented to guide it toward the target.

## Requirments
The following libraries are required to run BVRGym. 
The code has been tested with Python 3.9 

$ pip install jsbsim
$ pip install geopy
$ pip install pyproj
$ pip install pymap3d
# Choose own settings for pytorch
# More infor on https://pytorch.org/get-started/locally/
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ pip install tensorboard
$ pip install py_trees

## Getting started 
See Aircraft and missile performance 
$ python mainBVRGym.py -track f1 -head 0.0 -alt -1.0 -thr 1.0

Evading single missile 
# python mainBVRGym.py -track t1 -seed 1

Evading two missiles 
# python mainBVRGym.py -track t2 -seed 1

BVR air combat
# python mainBVRGym.py -track dog
