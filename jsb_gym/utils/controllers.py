
class PolicySelector():
    def __init__(self, ppo, env, k = 1, b = 500):
        self.V = {}
        self.V0 = {}
        self.CBF = {}
        self.ppo = ppo
        self.env = env
        self.k = k
        self.b = b

    def update_value_functions(self, state_block, state_block_old):
        for key in self.env.missile_block_names:
            self.V[key] =  self.env.inv_transform_reward(self.ppo.get_state_value(state_block[key])[0])
            self.V0[key] =  self.env.inv_transform_reward(self.ppo.get_state_value(state_block_old[key])[0])
    
    def update_CBF(self):
        for key in self.env.missile_block_names:
            self.CBF[key] = (self.V[key] - self.V0[key])/self.k + self.V[key] - self.b 
    
    def select_policy(self, state_block, state_block_old):
        self.update_value_functions(state_block, state_block_old)
        self.update_CBF()
        return min(zip(self.CBF.values(), self.CBF.keys()))[1]

class PID:
	"""
	Discrete PID control
	"""

	def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0, Integrator=0, Integrator_max=10, Integrator_min=-10):

		self.Kp=P
		self.Ki=I
		self.Kd=D
		self.Derivator=Derivator
		self.Integrator=Integrator
		self.Integrator_max=Integrator_max
		self.Integrator_min=Integrator_min

		self.set_point=0.0
		self.error=0.0

	def update(self,current_value):
		"""
		Calculate PID output value for given reference input and feedback
		"""

		self.error = self.set_point - current_value

		self.P_value = self.Kp * self.error
		self.D_value = self.Kd * ( self.error - self.Derivator)
		self.Derivator = self.error

		self.Integrator = self.Integrator + self.error

		if self.Integrator > self.Integrator_max:
			self.Integrator = self.Integrator_max
		elif self.Integrator < self.Integrator_min:
			self.Integrator = self.Integrator_min

		self.I_value = self.Integrator * self.Ki

		PID = self.P_value + self.I_value + self.D_value

		return PID

	def setPoint(self,set_point):
		"""
		Initilize the setpoint of PID
		"""
		self.set_point = set_point
		self.Integrator=0
		self.Derivator=0

	def setIntegrator(self, Integrator):
		self.Integrator = Integrator

	def setDerivator(self, Derivator):
		self.Derivator = Derivator

	def setKp(self,P):
		self.Kp=P

	def setKi(self,I):
		self.Ki=I

	def setKd(self,D):
		self.Kd=D

	def getPoint(self):
		return self.set_point

	def getError(self):
		return self.error

	def getIntegrator(self):
		return self.Integrator

	def getDerivator(self):
		return self.Derivator

