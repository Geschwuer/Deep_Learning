import numpy as np


# fan_in: how many input connections does one neuron have?
# fan_out: how many output connections does one neuron have?

class Constant:
    def __init__(self, const_val = 0.1):
        self.const_val = const_val
        
    def initialize(self, weights_shape, fan_in, fan_out):
        np.full(weights_shape, self.const_val)


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # create random values between 0.0 and 1.0 and use as output tensor the shape "weights_shape"
        return np.random.uniform(0.0, 1.0, size=weights_shape)
    

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # calculate sigma using Xavier formula
        sigma = np.sqrt(2/(fan_out + fan_in))
        # loc = mean (Mittelwert)
        # scale = sigma (Standardabweichung)
        return np.random.normal(loc = 0, scale= sigma, size=weights_shape)



class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        # calculate sigma using He formula
        sigma = np.sqrt(2/fan_in)
        # loc = mean (Mittelwert)
        # scale = sigma (Standardabweichung)
        return np.random.normal(loc = 0, scale= sigma, size=weights_shape)