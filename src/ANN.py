# imports
import numpy as np

# util functions

def linear(input, der=False):
    if not der+False:
        return input
    else:
        return 1

def relU(input, der=False):
    if not der:
        return max(0, np.round(input, 4))
    else:
        if input > 0:
            return 1
        else:
            return 0

def sigmoid(input, der=False):
    if not der:
        return np.round(1 / (1 + np.exp(-input)), 4)
    else:
        sigmoid_x = sigmoid(input, False)
        return np.round(sigmoid_x* (1 - sigmoid_x))

def h_tan(input, der=False):
    if not der:
        return np.round((np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input)), 4)
    else:
        return np.round((2/(np.exp(input) - np.exp(-input))**2), 4)

def zero_init(fr, to, params=None):
    return np.array([[0 for _ in range(fr)] for _ in range(to+1)]) #to + 1 for bias

def random_uniform(fr, to, params=None):
    np.random.seed(params["seed"])
    # return np.round(np.random.uniform(params["lb"], params["ub"], (fr*to)+2), fr*to).reshape((fr*to)+2, 1).reshape(fr+1, to)
    if params["seed"] == 42:
        return np.array([[0.35, 0.35], [0.15, 0.25], [0.20, 0.30]])
    if params["seed"] == 60:
        return np.array([[0.60, 0.60], [0.40, 0.50], [0.45, 0.55]])



# classes

class Layer:
    def __init__(self, a_func, w_init, neurons):
        """
        The configuration of a layer

        neurong: number of neurons in the layer
        a_func: activation function
        w_init: weight init function
        note: no weight yet, cause dimentions needs to be calculated when initiating
        weight also includes bias
        """
        self.neurons = neurons
        self.a_func = a_func
        self.w_init = w_init
        self.input = None
        self.error = np.array([])
        self.weight_to = None

class ANN:
    def __init__(self, data, config: list[Layer], input: Layer=None, output: Layer=None, load=None):
        """
        Initiation of an ANN

        config: array of Layer
        input: formatted input to Layer
        output: output size (integer)
        load: a formatted json input
        """
        self.data = data
        self.network: list[Layer] = [input] + config + [output]

        if not load:
            for i in range(len(self.network)-1):
                self.network[i].weight_to = self.network[i].w_init[0](self.network[i].neurons, self.network[i+1].neurons, self.network[i].w_init[1])
                # print(self.network[i].weight_to, "layer", i)
            # self.network[len(self.network)-1].weight_to = self.network[i].w_init[0](self.network[len(self.network)-1].neurons, self.network[i+1], self.network[len(self.network)-1].w_init[1])
            # print(self.network[len(self.network)-1].weight_to, "layer", len(self.network)-1)
        else:
            print(load)
            pass


    def train(self, b_size=None, l_rate=None, epoch=1, verb=None):
        """
        Trains the data using certain parameters and also shows the progress

        b_size: batch size
        l_rate: learning rate
        epoch: n numbers of epoch
        verb: to show progress bar (boolean)

        Returns the history of the training process
        """
        for _ in range(epoch):
            # forward propagation
            # print("forward prop")
            inp = np.array([[1, 0.05, 0.1]]) 
            self.network[0].input = inp
            # print(inp, "input")
            for i in range(len(self.network)-1):
                net = self.network[i].input @ self.network[i].weight_to
                # print(self.network[i].weight_to, "weight")
                if i == len(self.network)-2:
                    inp = self.network[i].a_func(net)
                else:
                    inp = np.concatenate((np.array([[1]]), self.network[i].a_func(net)), axis=1)
                # print(inp, "input")
                self.network[i+1].input = inp
            

            # backward propagation
            # print("backward prop")
            target = np.array([0.01, 0.99])
            out = self.network[len(self.network)-1].input
            self.network[len(self.network)-1].error = np.vectorize(self.error_translate)(out, target)

            for i in range(len(self.network)-2, 0, -1):
                weight = self.network[i].weight_to[1:]
                inp = self.network[i].input[0][1:]
                error = self.network[i+1].error[0]
                temp = []
                for j in range(len(inp)):
                    temp = np.append(self.network[i].error, np.dot(weight[j], error) * inp[j] * (1-inp[j]))
                    temp = np.array([temp])
                    # print(np.dot(weight[j], error), "(sigma error chain) *", inp[j], "(input) *", (1-inp[j]), "(1-input) calculation")
                    # print(self.network[i].error, "error")
                self.network[i].error = temp
            
            # Weight updating, including bias
            for i in range(len(self.network)-1):
                weight = self.network[i].weight_to
                # print(weight)
                for j in range(len(self.network[i].input[0])):
                    # print(weight[j])
                    # print(self.network[i+1].error)
                    # print(weight[j], (l_rate * self.network[i+1].error), self.network[i].input[0][j])
                    weight[j] = (weight[j] + (l_rate * self.network[i+1].error) * self.network[i].input[0][j])[0]
                self.network[i].weight_to = weight
            

    
    def show(self):
        """
        Shows the network structure
        """
        print("ANN structure:")
        for layer in self.network:
            print(layer.neurons, "neurons")
            print(layer.weight_to, "weight")
            print(layer.input, "input")
            print(layer.a_func, "activation function")
            print(layer.error, "error")
            print("")
        pass

    def error_translate(self, x, y):
        return (x * (1 - x) * (y - x))
    
    def weight_update(self, weight, l_rate, error, input):
        return weight + ((l_rate * error) * input)

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
layer_o = Layer(None, [None, {}], 2)
ann = ANN(None, [layer_1], input=layer_i, output=layer_o)
ann.train(l_rate=0.5, epoch=2)
ann.show()

