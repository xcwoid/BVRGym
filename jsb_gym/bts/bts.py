import py_trees as pt 
from jsb_gym.bts.reactive_seq import ReactiveSeq

from jsb_gym.bts.BVR.conditions import MAW_own_condition, MAW_condition, Pursue_condition, Launch_condition
from jsb_gym.bts.BVR.actions import MAW_guide_evade_action, MAW_evade_action, Guide_own_action, Pursue_action, Launch_action

from jsb_gym.bts.WVR.conditions import WVR_Pursue_condition
from jsb_gym.bts.WVR.actions import WVR_Pursue_action, WVR_Random_action


class BVRBT(object):
    def __init__(self, agent):
        self.agent = agent
        self.BTState = None
        self.BTState_old = None
        self.RootSuccess = False 
        self.root = ReactiveSeq("ReactiveSeq")
        self.use_memory = False

        '''Missile awerness system MAW'''        
        self.MAW_own = pt.composites.Selector(name = "13", memory = self.use_memory)
        self.MAW_own_con = MAW_own_condition('13C', self.agent)
        self.MAW_guide_evade_act = MAW_guide_evade_action('13A', self.agent)
        self.MAW_own.add_children([self.MAW_own_con, self.MAW_guide_evade_act])
        
        self.MAW2 = pt.composites.Sequence(name = "12", memory = self.use_memory) #2
        self.MAW_evade_act = MAW_evade_action('12A', self.agent)
        self.MAW2.add_children([self.MAW_own, self.MAW_evade_act])

        self.MAW = pt.composites.Selector(name = "11", memory = self.use_memory) # 1
        self.MAW_con = MAW_condition('11C', self.agent)
        self.MAW.add_children([self.MAW_con, self.MAW2])

        '''Missile guidance'''
        self.guide = pt.composites.Selector(name = "21", memory = self.use_memory) # 1
        self.guide_own_con = MAW_own_condition('21C', self.agent)
        self.guide_own_act = Guide_own_action('21A', self.agent)
        self.guide.add_children([self.guide_own_con, self.guide_own_act])

        '''launch'''
        self.launch = pt.composites.Selector(name = "31", memory = self.use_memory) # 1
        self.launch_con = Launch_condition('31C', self.agent)
        self.launch_act = Launch_action('31A', self.agent)
        self.launch.add_children([self.launch_con, self.launch_act])

        '''pursue'''
        self.pursue = pt.composites.Selector(name= "41", memory = self.use_memory) # 1
        self.pursue_con = Pursue_condition('41C', self.agent)
        self.pursue_act = Pursue_action('41A', self.agent)
        self.pursue.add_children([self.pursue_con, self.pursue_act])

        '''root'''
        self.root.add_children([self.MAW, self.guide, self.launch, self.pursue])
        #tree = pt.trees.BehaviourTree(self.root)
        #print(ascii_tree(self.root))

    def tick(self):
        #print('-'*10)
        self.root.tick_once()
        self.BTState = self.root.tip().name
        #print('-')
        #if self.BTState != self.BTState_old:
        if self.BTState == '13A':
            self.heading = self.MAW_guide_evade_act.heading
            self.altitude = self.MAW_guide_evade_act.altitude
            self.launch_missile = self.MAW_guide_evade_act.launch_missile
        elif self.BTState == '12A':
            self.heading = self.MAW_evade_act.heading
            self.altitude = self.MAW_evade_act.altitude
            self.launch_missile = self.MAW_evade_act.launch_missile
        elif self.BTState == '21A':
            self.heading = self.guide_own_act.heading
            self.altitude = self.guide_own_act.altitude
            self.launch_missile = self.guide_own_act.launch_missile
        elif self.BTState == '31A':
            self.heading = self.launch_act.heading
            self.altitude = self.launch_act.altitude
            self.launch_missile = self.launch_act.launch_missile
        elif self.BTState == '41A':
            self.heading = self.pursue_act.heading
            self.altitude = self.pursue_act.altitude
            self.launch_missile = self.pursue_act.launch_missile
        else:
            print('Unexpected state')
            exit()

        self.BTState_old = self.BTState



class WVRBT(object):
    def __init__(self, agent):
        self.agent = agent
        self.BTState = None
        self.BTState_old = None
        self.RootSuccess = False 
        self.root = ReactiveSeq("ReactiveSeq")
        self.use_memory = False

        '''pursue'''
        self.pursue = pt.composites.Selector(name= "11", memory = self.use_memory) # 1
        self.pursue_con = WVR_Pursue_condition('11C', self.agent)
        self.pursue_act = WVR_Pursue_action('11A', self.agent)
        self.pursue.add_children([self.pursue_con, self.pursue_act])

        '''root'''
        self.root.add_children([self.pursue])
        #tree = pt.trees.BehaviourTree(self.root)
        #print(ascii_tree(self.root))

    def tick(self):
        self.root.tick_once()
        self.BTState = self.root.tip().name
        if self.BTState == '11A':
            self.heading = self.pursue_act.heading
            self.altitude = self.pursue_act.altitude
        else:
            print('Unexpected state')
            exit()

        self.BTState_old = self.BTState


class RandomBT(object):
    def __init__(self, agent):
        self.agent = agent
        self.BTState = None
        self.BTState_old = None
        self.RootSuccess = False 
        self.root = ReactiveSeq("ReactiveSeq")
        self.use_memory = False

        '''pursue'''
        self.pursue = pt.composites.Selector(name= "11", memory = self.use_memory) # 1
        self.pursue_con = WVR_Pursue_condition('11C', self.agent)
        self.pursue_act = WVR_Random_action('11A', self.agent)
        self.pursue.add_children([self.pursue_con, self.pursue_act])

        '''root'''
        self.root.add_children([self.pursue])
        #tree = pt.trees.BehaviourTree(self.root)
        #print(ascii_tree(self.root))

    def tick(self):
        self.root.tick_once()
        self.BTState = self.root.tip().name
        if self.BTState == '11A':
            self.heading = self.pursue_act.heading
            self.altitude = self.pursue_act.altitude
        else:
            print('Unexpected state')
            exit()

        self.BTState_old = self.BTState










