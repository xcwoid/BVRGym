# BVRGym
## Description
This library is heavily based on JSBSim software (https://github.com/JSBSim-Team/jsbsim). 
This library's primary purpose is to allow users to explore Beyond Visual Range (BVR) tactics using Reinforcement learning.

## Environment
Currently, there are three available environments:
Evading one missile 
Evading two missile 
BVR air combat
The environments above mainly use the F16 flight dynamics and BVR missile models. 
The F16 model has an additional wrapper to control simply, while the BVR missile has a Proportional Navigation guidance law implemented to guide it toward the target.

## Requirments
The following libraries are required to run BVRGym. 
The code has been tested with Python 3.9 

pip install jsbsim geopy pyproj pymap3d torch tensorboard py_trees

## Getting started 
See Aircraft and missile performance 
$ python mainBVRGym.py -track f1 -head 0.0 -alt -1.0 -thr 1.0

### Evading single missile 
python mainBVRGym.py -track t1 -seed 1

### Evading two missiles 
python mainBVRGym.py -track t2 -seed 1

### BVR air combat
python mainBVRGym.py -track dog
