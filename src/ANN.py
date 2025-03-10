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

    def train(self, b_size, l_rate, epoch, verb):
        """
        Trains the data using certain parameters and also shows the progress

        b_size: batch size
        l_rate: learning rate
        epoch: n numbers of epoch
        verbose: to show progress (boolean)

        Returns the history of the training process
        """
        pass
    
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
        pass
    
    def save(self):
        """
        Saves the model instance using a format (json?)
        """
        pass

    def load(self):
        """
        Loads a model instance using a format
        """
        pass


