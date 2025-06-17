class BaseLayer:
    def __init__(self):
        self.trainable: bool = False
        self.weights = None