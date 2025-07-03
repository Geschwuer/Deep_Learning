class BaseLayer:
    def __init__(self):
        self.trainable: bool = False
        self.testing_phase = False