# BVRGym
## Description
This library is based on JSBSim software (https://github.com/JSBSim-Team/jsbsim). 
This library's primary purpose is to allow users to explore Beyond Visual Range (BVR) tactics using Reinforcement learning while utilizing JSBSim high-fidelity flight dynamics models.

## Requirments
The following libraries are required to run BVRGym. 
The code has been tested with Python 3.11 

`pip install gymnasium jsbsim pymap3d pandas py_trees stable_baselines3 tensorboard`

## BVRGym

BVRGym has a 3-level structure. Level 1 is the lowest level where Python interacts with the JSBSim flight dynamics simulator. Examples of Level 1 objects are located in the BVRGYM/jsb_gym/simObjects directory. Here we have both the aircraft class and the missile class. Level 2 represents agents, a wrapper that has both missiles and aircraft objects, i.e., they use level 1 objects. And level 3 is the environments that use agents. Each level has its own configuration files. 

To test these objects in an isolated scenario, install Flightgear (https://www.flightgear.org/), place yourself in the BVRGYM directory, and then run commands as explained below


### FlightGear
FlightGear offers an excellent tool for visualizing the units present in BVRGym. 

`sudo add-apt-repository ppa:saiarcot895/flightgear`

`sudo apt update`

More details on https://launchpad.net/~saiarcot895/+archive/ubuntu/flightgear

After installing FlightGear, start the program, go to the Aircraft tab, and download the General Dynamics F16 aircraft.

To have a missile model as well, copy the ogel folder from this repo (BVRGYM/fg/ogel) to the Flightgears aircraft directory.
To visualize a missile in FlightGear, I used the ogel (1) model within FGFS, replaced the graphical representations of the ogel with a missile available in FGFS (I think it was from (2)), and added a trail from Santa Claus (3) to see the trajectory. 

1) https://wiki.flightgear.org/Ogel

2) https://forum.flightgear.org/viewtopic.php?t=19930#p183249

3) https://wiki.flightgear.org/Santa_Claus

Given that you have both the f16 and ogel directories in the fgfs directory (home/.fgfs/Aircraft/...), you will be able to visualize both units in the next steps

<img width="1268" height="483" alt="fgfs" src="https://github.com/user-attachments/assets/67e3d59e-035d-4ef7-93d1-6ab9aa1950c6" />

### To test aircraft (Level 1)

terminal 1 (Visualize F16):

`fgfs --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft=f16-block-52 --airport=ESSA --multiplay=out,10,127.0.0.1,5000 --multiplay=in,10,127.0.0.1,5001`
 

https://github.com/user-attachments/assets/3fcf87ff-1620-43b0-a872-0158c619194b


terminal 2:

`
python -m tests.test_aircraft
`


https://github.com/user-attachments/assets/ee11d1e6-7404-44f1-8df2-4e457a551fc2


### To test missile (Level 1)

terminal 1: 

`fgfs --fdm=null --native-fdm=socket,in,60,,5550,udp --aircraft=f16-block-52 --airport=ESSA --multiplay=out,10,127.0.0.1,5000 --multiplay=in,10,127.0.0.1,5001`

terminal 2: 

`fgfs --fdm=null --native-fdm=socket,in,60,,5551,udp --aircraft=ogel --airport=ESSA --multiplay=out,10,127.0.0.1,5001 --multiplay=in,10,127.0.0.1,5000`

terminal 3:
`
python -m tests.test_missile
`


https://github.com/user-attachments/assets/edada4ad-bd02-46a3-81dc-c766005f0803


### Test Agent (Level 2)

Check that all the ammo is loaded.

`
python -m tests.test_agent
`

### Test Tacview (Level 3)

RL agent takes random actions, while the red aircraft is controlled by a behavior tree.

`
python -m tests.test_tacview
`

Open tacview (https://www.tacview.net/) and load data in BVRGym/data_output/tacview

https://github.com/user-attachments/assets/23e829d6-33c3-4b63-b559-539241c85105

## BVR air combat training

To start training an agent for BVR air combat, run the following command

`
python main.py 
`



## Additional details 
Additional details can be found in the following article

BVR Gym: A Reinforcement Learning Environment for Beyond-Visual-Range Air Combat

https://arxiv.org/abs/2403.17533

Note that this article relates to an older version of the BVRGym, but it is still valid with remarks to understand Behavior Trees and other part of the environments.


## BVRGym for WVR air combat

Even though this library focuses on BVR (beyond-visual-range) air combat, WVR (within-visual-range) combat remains a critical aspect of aerial warfare, and pilots continue to train extensively for it. One of my favorite articles in this field is https://arxiv.org/pdf/2105.00990, where the authors applied hierarchical reinforcement learning (HRL) to WVR air combat scenarios.

In this section, I will reimplement parts of that work; however, the results will not match the performance reported in the original study. The goal here is to keep the implementation as lightweight and simple as possible while still capturing the core ideas.


`
python -m tests.test_wvr
`

### Control Zone, AggressiveShooter or ConservativeShooter training
Uncomment for example train_WvrControlZone() in the main file and run 
`
python main.py 
`

# Self-play
TODO

# HRL

TODO