import numpy as np

class Constant:
    def __init__(self, value = 0.1):
        self.value = value
    
    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.full(weights_shape, self.value) # Returns a constant array of the specified shape
        
         
class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.rand(*weights_shape) # * to "unpack" tuple into several arguments
        
class Xavier:
    def init(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # calculate sigma using Xavier formula
        sigma = np.sqrt(2/(fan_out + fan_in))
        # loc = mean (Mittelwert)
        # scale = sigma (Standardabweichung)
        return np.random.normal(loc = 0, scale= sigma, size=weights_shape)

class He:
    def init(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # calculate sigma using He formula
        sigma = np.sqrt(2/fan_in)
        # loc = mean (Mittelwert)
        # scale = sigma (Standardabweichung)
        return np.random.normal(loc = 0, scale= sigma, size=weights_shape)