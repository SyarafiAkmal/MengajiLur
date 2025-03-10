#imports

class Layer:
    def __init__(self, neurons: int, a_func, w_init):
        """
        The configuration of a layer

        neurons: n of neurons
        a_func: activation function
        w_init: weight init method
        """
        self.neurons = neurons
        self.a_func = a_func
        self.w_init = w_init

class ANN:
    def __init__(self, config: list[Layer]):
        """
        Initiation of an ANN

        config: array of Layers
        """
        self.config = config
    
    def show(self):
        """
        Shows the network structure
        """
        pass

    def w_dist_show(self):
        """
        Shows the weight distribution of each layer
        """
        pass

    def wg_dist_show(self):
        """
        Shows the weight gradient distribution of each layer
        """
    
    def save(self):
        """
        Saves the model instance using a format (json?)
        """
        pass

    def load(self):
        """
        Loads a model instance using a format
        """

