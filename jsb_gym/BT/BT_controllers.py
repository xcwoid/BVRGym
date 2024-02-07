import py_trees as pt 
from jsb_gym.BT.reactive_seq import ReactiveSeq
import numpy as np 
from jsb_gym.tools import toolkit

class DogFight_BT(object):
    def __init__(self, env, fdm, fdm_tgt):
        self.BTState = None
        self.BTState_old = None
        self.RootSuccess = False 
        self.root = ReactiveSeq("ReactiveSeq")
        
        '''Altitude safety'''
        self.Altitude_Safety = pt.composites.Selector("AltitudeSafety")
        self.Alt_con = Altitude_Check('Altitude_Check', fdm.fdm)
        self.Alt_act = Altitude_Control('Altitude_Control', fdm.fdm)
        self.Altitude_Safety.add_children([self.Alt_con, self.Alt_act])

        '''Control Zone'''
        self.Control_Zone = pt.composites.Selector("ControlZone")
        self.CZ_con = CZ_Check('CZ_Check', env)
        self.CZ_act = CZ_Control('CZ_Control', env)
        self.Control_Zone.add_children([self.CZ_con, self.CZ_act])

        '''Aim at the target'''
        self.Aim = pt.composites.Selector("Aim")
        self.Aim_con = Aim_Check('Aim_Check', env= env, fdm= fdm, fdm_tgt= fdm_tgt)
        self.Aim_act = Aim_Control('Aim_Control', env= env, fdm= fdm)
        self.Aim.add_children([self.Aim_con, self.Aim_act])


        self.root.add_children([self.Altitude_Safety, self.Aim])

    def tick(self, show_state = False):
        self.root.tick_once()
        self.BTState = self.root.tip().name
        if show_state:
            if self.BTState != self.BTState_old:
                print(self.BTState)
        self.BTState_old = self.BTState
        
class DogFight_RLBT(object):
    def __init__(self, env, fdm, fdm_tgt, show_state = False):
        self.show_state = show_state
        self.BTState = None
        self.BTState_old = None
        self.RootSuccess = False 
        self.root = ReactiveSeq("ReactiveSeq")
        
        '''Altitude safety'''
        self.Altitude_Safety = pt.composites.Selector("AltitudeSafety")
        self.Alt_con = Altitude_Check('Alt_con', fdm.fdm)
        self.Alt_act = Altitude_Control('Alt_act', fdm.fdm)
        self.Altitude_Safety.add_children([self.Alt_con, self.Alt_act])

        '''Dummy branch'''
        self.dummy = pt.composites.Selector("Dummy")
        self.dum_con = dum_con('dum_con', fdm= fdm)
        self.dum_act = dum_act('dum_act', fdm= fdm)
        self.dummy.add_children([self.dum_con, self.dum_act])

        self.root.add_children([self.Altitude_Safety, self.dummy])

    def tick(self):
        self.root.tick_once()
        self.BTState = self.root.tip().name
        if self.show_state:
            if self.BTState != self.BTState_old:
                print(self.BTState)
        self.BTState_old = self.BTState

class Aim_Check(pt.behaviour.Behaviour):
    def __init__(self, name, env, fdm, fdm_tgt):
        super(Aim_Check, self).__init__(name)
        self.env = env
        self.fdm = fdm 
        self.fdm_tgt = fdm_tgt
        
               
    def update(self):
        self.feedback_message = "Aim_Check"
        if False:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class Aim_Control(pt.behaviour.Behaviour):
    def __init__(self, name, env, fdm):
        super(Aim_Control, self).__init__(name)
        self.env = env
        self.fdm = fdm 
        self.v1 = np.array([1.0, 0.0, 0.0])
        self.v2 = np.array([0.0, 0.0, 0.0])
        self.tk = toolkit()
        self.psi_ref = None
        self.theta_ref = None

    def update(self):
        self.feedback_message = "Aim_Control"
        '''Get state in the N,E,Up frame of the agents position'''
        self.v2[0] = self.fdm.state[0,5]
        self.v2[1] = self.fdm.state[0,6]
        '''
        Get angle in the North East coordinate frame.
        [N, E, Up]
        '''
        self.psi_ref = self.tk.angle_between(self.v1, self.v2)
        
        '''
        Get direction of the angle 
        +1 means N -> E rotation 
        -1 N -> W rotation  
        '''
        psi_direction = np.cross(self.v1, self.v2)
        if psi_direction[2] < 0.0:
            self.psi_ref = -self.psi_ref
        
        '''Heading reference in radians'''
        self.psi_ref = self.tk.translate_semi_to_full_circle(self.psi_ref)
        
        '''Add altitude component'''
        alt = self.fdm.state[0,7]
        xy = np.linalg.norm(self.v2)
        
        '''Pitch in radians '''
        self.theta_ref = np.arctan2(alt, xy)

        return pt.common.Status.RUNNING

class CZ_Check(pt.behaviour.Behaviour):
    def __init__(self, name, env):
        super(CZ_Check, self).__init__(name)
        self.env = env
          
    def update(self):
        self.feedback_message = "CZ_Check"
        '''Check if close enough'''        
        if self.env.separation != None and self.env.separation <=  3000.0:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class CZ_Control(pt.behaviour.Behaviour):
    def __init__(self, name, env):
        super(CZ_Control, self).__init__(name)
        self.env = env
        self.v1 = np.array([1.0, 0.0, 0.0])
        self.v2 = np.array([0.0, 0.0, 0.0])
        self.tk = toolkit()
        self.psi_ref = None
        self.theta_ref = None


    def update(self):
        self.feedback_message = "CZ_Control"
        self.separation = self.env.separation
        self.altitude_difference = self.env.f16.fdm['position/h-sl-meters'] - self.env.f16r.fdm['position/h-sl-meters']
        self.altitude_ref = self.env.f16.fdm['position/h-sl-meters']
        
        '''Get state in the N,E,Up frame of the agents position'''
        self.v2[0] = self.env.state_r[0,5]
        self.v2[1] = self.env.state_r[0,6]
        
        '''
        Get angle in the North East coordinate frame.
        [N, E, Up]
        '''
        self.psi_ref = self.tk.angle_between(self.v1, self.v2)
        
        '''
        Get direction of the angle 
        +1 means N -> E rotation 
        -1 N -> W rotation  
        '''
        psi_direction = np.cross(self.v1, self.v2)
        if psi_direction[2] < 0.0:
            self.psi_ref = -self.psi_ref
        
        '''Heading reference in radians'''
        self.psi_ref = self.tk.translate_semi_to_full_circle(self.psi_ref)
        
        '''Add altitude component'''
        alt = self.env.state[0,7]
        xy = np.linalg.norm(self.v2)
        
        '''Pitch in radians '''
        self.theta_ref = np.arctan2(alt, xy)
        

        return pt.common.Status.RUNNING

class Altitude_Check(pt.behaviour.Behaviour):
    def __init__(self, name, fdm):
        super(Altitude_Check, self).__init__(name)
        self.fdm = fdm
        
               
    def update(self):
        self.feedback_message = "Altitude_Check"
        alt = self.fdm['position/h-sl-meters']
        if alt >= 2000.0:
            # if above 2000 m return success
            return pt.common.Status.SUCCESS
        else:
            # if below 2000 m return failure 
            return pt.common.Status.FAILURE

class Altitude_Control(pt.behaviour.Behaviour):
    def __init__(self, name, fdm):
        super(Altitude_Control, self).__init__(name)
        self.fdm = fdm


    def update(self):
        self.feedback_message = "Altitude_Control"
        return pt.common.Status.RUNNING

class dum_con(pt.behaviour.Behaviour):
    def __init__(self, name, fdm):
        super(dum_con, self).__init__(name)
        self.fdm = fdm
        
               
    def update(self):
        self.feedback_message = "dum_con"
        if False:
            # if above 2000 m return success
            return pt.common.Status.SUCCESS
        else:
            # if below 2000 m return failure 
            return pt.common.Status.FAILURE

class dum_act(pt.behaviour.Behaviour):
    def __init__(self, name, fdm):
        super(dum_act, self).__init__(name)
        self.fdm = fdm

    def update(self):
        self.feedback_message = "dum_act"
        return pt.common.Status.RUNNING
