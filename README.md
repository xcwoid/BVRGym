# BVRGym
## Description
This library is heavily based on JSBSim software (https://github.com/JSBSim-Team/jsbsim). 
This library's primary purpose is to allow users to explore Beyond Visual Range (BVR) tactics using Reinforcement learning.

![me](https://github.com/xcwoid/BVRGym/blob/main/fg_git.gif)

## Environment
The environments above mainly use the F16 flight dynamics and BVR missile models. 
The F16 model has an additional wrapper to control simply, while the BVR missile has a Proportional Navigation guidance law implemented to guide it toward the target.
Currently, there are three available environments:

### Evading one missile 

### Evading two missile 

### BVR air combat
At the beginning of the training, we see that the Red aircraft effectively shoots down the agent with its first missile (aim1r) but later starts using the second missile as well (aimr2). As the training progresses, the agent starts to utilize their own missiles (aim1) and (aim2), and the running reward illustrates that the agent slowly improves its behavior towards defeating the enemy.  

![me](https://github.com/xcwoid/BVRGym/blob/main/BVRGymTraining_git.png)


## Requirments
The following libraries are required to run BVRGym. 
The code has been tested with Python 3.9 

pip install jsbsim geopy pyproj pymap3d torch tensorboard py_trees

## Getting started 
To plot Aircraft and Missile behavior 

python mainBVRGym.py -track f1 -head 0.0 -alt -1.0 -thr 1.0

### Evading single missile 
python mainBVRGym.py -track t1 -seed 1

### Evading two missiles 
python mainBVRGym.py -track t2 -seed 1

### BVR air combat
python mainBVRGym.py -track dog

## Parallel processing 
The following commands are equivalent to the ones above, but they run the process in parallel to speed up convergence. 

### Evading single missile 
python mainBVRGym_MultiCore.py -track M1  -cpus 8 -Eps 100000 -eps 3

### Evading two missiles 
python mainBVRGym_MultiCore.py -track M2  -cpus 8 -Eps 100000 -eps 3

### BVR air combat
python mainBVRGym_MultiCore.py -track Dog -cpus 8 -Eps 10000 -eps 1
