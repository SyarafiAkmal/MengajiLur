# imports
import numpy as np

# util functions

def linear(input):
    return input

def relU(input):
    if input < 0:
        input = 0
    return np.round(input, 4)

def sigmoid(input):
    return np.round(1 / (1 + np.exp(-input)), 4)

def h_tan(input):
    return np.round((np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input)), 4)

def zero_init(fr, to, params=None):
    return np.array([[0 for _ in range(fr)] for _ in range(to+1)]) #to + 1 for bias

def random_uniform(fr, to, params=None):
    np.random.seed(params["seed"])
    return np.round(np.random.uniform(params["lb"], params["ub"], (fr*to)+2), fr*to).reshape((fr*to)+2, 1).reshape(fr+1, to)



# classes

class Layer:
    def __init__(self, a_func, w_init, neurons):
        """
        The configuration of a layer

        a_func: activation function
        w_init: weight init method
        note: no weight yet, cause dimentions needs to be calculated when initiating
        weight also includes bias
        """
        self.neurons = neurons
        self.a_func = a_func
        self.w_init = w_init
        self.weight_to = None

class ANN:
    def __init__(self, data, config: list[Layer], input=None, output=None, load=None):
        """
        Initiation of an ANN

        config: array of Layer
        input: formatted input to Layer
        output: output size
        """
        self.data = data
        self.output = output
        self.network: list[Layer] = [input] + config

        if not load:
            for i in range(len(self.network)-1):
                self.network[i].weight_to = self.network[i].w_init[0](self.network[i].neurons, self.network[i+1].neurons, self.network[i].w_init[1])
                print(self.network[i].weight_to, "layer", i)
            self.network[len(self.network)-1].weight_to = self.network[i].w_init[0](self.network[len(self.network)-1].neurons, self.output, self.network[len(self.network)-1].w_init[1])
            print(self.network[len(self.network)-1].weight_to, "layer", len(self.network)-1)
        else:
            print(load)
            pass


    def train(self, b_size=None, l_rate=None, epoch=None, verb=None):
        """
        Trains the data using certain parameters and also shows the progress

        b_size: batch size
        l_rate: learning rate
        epoch: n numbers of epoch
        verb: to show progress bar (boolean)

        Returns the history of the training process
        """
        # forward propagation
        # inp = np.array([[1], [0.05], [0.1]])
        # wi_1 = np.array([[0.35, 0.35], [0.15, 0.25], [0.20, 0.30]])
        # wi_1_t = wi_1.T

        # # print(w_t)
        # # print(inp)
        # layer_1 = wi_1_t @ inp
        # layer_1 = sigmoid(layer_1)
        # layer_1 = np.concatenate((np.array([[1]]), layer_1), axis=0)
        # print(layer_1)

        # w1_2 = np.array([[0.60, 0.60], [0.40, 0.50], [0.45, 0.55]])
        # w1_2_t = w1_2.T

        # output = w1_2_t @ layer_1
        # output = sigmoid(output)
        inp = np.array([[1], [0.05], [0.1]])
        print(inp)
        for layer in self.network:
            net = layer.weight_to.T @ inp
            inp = np.concatenate((np.array([[1]]), layer.a_func(net)), axis=0)
            print(inp)
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

layer_i = Layer(sigmoid, [random_uniform, {"seed": 42, "lb": 0, "ub": 0.5}], 2)
layer_1 = Layer(sigmoid, [random_uniform, {"seed": 60, "lb": 0, "ub": 0.5}], 2)
layer_2 = Layer(sigmoid, [random_uniform, {"seed": 70, "lb": 0, "ub": 0.5}], 2)
ann = ANN(None, [layer_1], input=layer_i, output=2)
ann.train()

