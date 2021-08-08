class Agent(object):
    def __init__(self, name, ind, observation_space, action_space):
        self.name = name
        self.index = ind
        self.observation_space = observation_space
        self.action_space = action_space

