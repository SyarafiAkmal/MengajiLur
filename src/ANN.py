# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
import networkx as nx

# activation functions

def linear(input, der=False):
    if not der+False:
        return input
    return 1

def relU(input, der=False):
    if not der:
        return np.maximum(0, input)
    return np.where(input > 0, 1, 0)

def eLU(input, der=False):
    if not der:
        return 

def sigmoid(input, der=False):
    if not der:
        return 1 / (1 + np.exp(-input))
    return input * (1 - input)

def h_tan(input, der=False):
    if not der:
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
    return (2/(np.exp(input) - np.exp(-input))**2)

def softmax(input, der=False):
    if not der:
        exp_x = np.exp(input - np.max(input))
        return exp_x / np.sum(exp_x)
    
def step_function(input, der=False):
    if der:
        return 0
    return np.where(input >= 0, 1, 0)

# error loss functions

def SSE(target, pred, der=False):
    if not der:
        return 0.5 * np.sum((target - pred) ** 2)
    return target - pred

def MSE(target, pred, der=False): # if multiple neuron, enter the mean as pred
    if not der:
        return -2*np.mean(target-pred)
    return np.mean((target - pred) ** 2)

# print(MSE(np.array([1.5, 0.1, 1.2]), np.array([1.3, 0.09, 1.3])))

def BCE(target, pred):
    return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

# print(BCE(np.array([0.5, 0.4, 0.9]), np.array([0.6, 0.5, 0.7])))

def CCE(target, pred):
    return -np.mean(np.sum(target * np.log(pred), axis=1))

# print(CCE(np.array([
#     [0, 1, 0],
#     [1, 0, 0],
#     [0, 0, 1]
# ]), np.array([
#     [0.1, 0.7, 0.2],
#     [0.8, 0.1, 0.1],
#     [0.3, 0.2, 0.5]
# ])))

# weight init functions

def zero_init(fr, to, params=None):
    return np.array([[0 for _ in range(fr)] for _ in range(to+1)]) #to + 1 for bias

def random_uniform(fr, to, params=None):
    np.random.seed(params["seed"])
    # print(fr, to)
    return np.round(np.random.uniform(params["lb"], params["ub"], (fr*to)+to), fr*to).reshape((fr*to)+to, 1).reshape(fr+1, to)
    # if params["seed"] == 42: # for debug purpose
    #     return np.array([[0, -1], [1, 1], [1, 1]])
    #     return np.array([[0.35, 0.35], [0.15, 0.25], [0.20, 0.30]])
    # if params["seed"] == 60:
    #     return np.array([[0], [1], [-2]])
    #     return np.array([[0.6, 0.6], [0.4, 0.5], [0.45, 0.55]])



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
        self.weight_grad = None 

class ANN:
    def __init__(self, data, config: list[Layer], input: Layer=None, output: Layer=None, error=None, load=None):
        """
        Initiation of an ANN

        config: array of Layer
        input: formatted input to Layer
        output: output size (integer)
        load: a formatted json input
        """
        self.data = data
        self.network: list[Layer] = [input] + config + [output]
        self.err_func = error

        if not load:
            for i in range(len(self.network)-1):
                self.network[i].weight_to = self.network[i].w_init[0](self.network[i].neurons, self.network[i+1].neurons, self.network[i].w_init[1])
                # print(self.network[i].weight_to, "layer", i)
            # self.network[len(self.network)-1].weight_to = self.network[i].w_init[0](self.network[len(self.network)-1].neurons, self.network[i+1], self.network[len(self.network)-1].w_init[1])
            # print(self.network[len(self.network)-1].weight_to, "layer", len(self.network)-1)
        else:
            print(load)
            pass


    def train(self, b_size=None, l_rate=None, epoch=1, verb=None): # NOTE: bsize no need i think, just input of batch input configured matrix
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
            # inp = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  
            self.network[0].input = inp
            # print(inp, "input")
            for i in range(len(self.network)-1):
                net = self.network[i].input @ self.network[i].weight_to
                # print(self.network[i].weight_to, "weight")
                if i == len(self.network)-2:
                    inp = self.network[i].a_func(net)
                else:
                    res = self.network[i].a_func(net)
                    inp = np.concatenate((np.ones((res.shape[0], 1)), res), axis=1)
                    # print(inp)
                # print(inp, "input")
                self.network[i+1].input = inp
            

            # backward propagation
            # print("backward prop")
            # target = np.array([[0], [1], [1], [0]])
            target = np.array([[0.01, 0.99]])
            out = self.network[len(self.network)-1].input
            self.network[len(self.network)-1].error = np.array([np.mean(np.vectorize(self.error_translate)(self.network[len(self.network)-2].a_func, out, target), axis=0)])

            for i in range(len(self.network)-2, 0, -1):
                weight = self.network[i].weight_to[1:]
                inp = self.network[i].input[0][1:]
                error = self.network[i+1].error[0]
                temp=np.array([])
                # print(self.network[i].input, "inp")
                for j in range(len(inp)):
                    temp = np.append(temp, np.dot(weight[j], error) * self.network[i].a_func(inp[j], True))
                    temp = np.array([temp])
                    # print(np.dot(weight[j], error), "(sigma error chain) *", inp[j], "(input) *", (1-inp[j]), "(1-input) calculation")
                    # print(self.network[i].error, "error")
                self.network[i].error = temp
            
            # Weight updating, including bias
            for i in range(len(self.network)-1):
                weight = self.network[i].weight_to
                # print(weight)
                temp=[]
                for j in range(len(self.network[i].input[0])):
                    # print(weight[j], "weight")
                    # print((l_rate * self.network[i+1].error) * self.network[i].input[0][j])
                    # print(self.network[i].input[0][j], "input")
                    temp += [(weight[j] + (l_rate * self.network[i+1].error) * self.network[i].input[0][j])[0]]
                self.network[i].weight_to = np.array(temp)
                # print(self.network[i].weight_to)
            

    
    def show(self):
        """
        Shows the network structure
        """
        print("ANN structure:")
        for index, layer in enumerate(self.network):
            if index == 0:
                print("input layer")
            elif index == len(self.network) - 1:
                print("output layer")
            else:
                print("hidden layer")
            print(layer.neurons, "neurons")
            print(layer.weight_to, "weight")
            print(layer.input, "input")
            print(layer.a_func, "activation function")
            print(layer.error, "error term")
            print("")
        
        if self.network[len(self.network)-1].input is not None:
            print(self.err_func(np.array([[0.01, 0.99]]), self.network[len(self.network)-1].input), "error")
        else:
            print("Network not initialized")

    def error_translate(self, a_func, x, y):
        # x * (1 - x): derivative of the a_func
        # (y - x): derivative of the loss func 
        return (a_func(x, True) * self.err_func(y, x, True))
    
    def weight_update(self, weight, l_rate, error, input):
        return weight + ((l_rate * error) * input)
    
    # There is something wrong with the seeding, must revisit later
    def w_dist_show(self, layers=None):   
        if layers is None:
            layers = []
            for i in range(len(self.network)):
                if hasattr(self.network[i], 'weight_to') and self.network[i].weight_to is not None:
                    layers.append(i)
                    # print(f"layer {i}")
        
        good_layers = []
        for idx in layers:
            if idx >= 0 and idx < len(self.network):
                net = self.network[idx]
                if hasattr(net, 'weight_to') and net.weight_to is not None:
                    good_layers.append(idx)
                else:
                    print(f"Layer {idx} have no weights.")
            else:
                print(f"Layer {idx} out of range.")
        
        if len(good_layers) == 0:
            print("No correct layers to visualize.")
            return
        
        fig, axes = plt.subplots(1, len(good_layers), figsize=(6*len(good_layers), 5))
        
        if len(good_layers) == 1:
            axes = [axes]
        
        i = 0
        for layer_idx in good_layers:
            w = self.network[layer_idx].weight_to.flatten()
            
            axes[i].hist(w, bins=20, color='blue', alpha=0.7)
            axes[i].set_title(f'Layer {layer_idx} Weight Distribution')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            i += 1
        
        plt.tight_layout()
        plt.show()
        
    def wg_dist_show(self, layers=None):
        if layers is None:
            layers = []
            for i in range(len(self.network)):
                if hasattr(self.network[i], 'weight_grad') and self.network[i].weight_grad is not None:
                    layers.append(i)
        
        valid_layer_list = []
        for l_idx in layers:
            layer_valid = False
            if l_idx >= 0 and l_idx < len(self.network):
                layer = self.network[l_idx]
                if hasattr(layer, 'weight_grad') and layer.weight_grad is not None:
                    valid_layer_list.append(l_idx)
                    layer_valid = True
            
            if not layer_valid:
                print(f"Layer {l_idx} has no gradients or invalid.")
        
        if len(valid_layer_list) == 0:
            print("No correct layers to visualize.")
            return
        
        fig, ax_array = plt.subplots(1, len(valid_layer_list), figsize=(6*len(valid_layer_list), 5))
        
        if len(valid_layer_list) == 1:
            ax_array = [ax_array]
        
        for plot_idx in range(len(valid_layer_list)):
            l_idx = valid_layer_list[plot_idx]
            grads = self.network[l_idx].weight_grad.flatten()
            
            ax_array[plot_idx].hist(grads, bins=20, color='red', alpha=0.7)
            ax_array[plot_idx].set_title(f'Layer {l_idx} Weight Gradient Distribution')
            ax_array[plot_idx].set_xlabel('Gradient Value')
            ax_array[plot_idx].set_ylabel('Frequency')
            ax_array[plot_idx].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def visualize_network(self, show_weights=True, show_gradients=False, filename=None): 
        graph = nx.DiGraph()
        positions = {}
        labels = {}
        
        for layer_idx, layer in enumerate(self.network):
            for neuron_idx in range(layer.neurons):
                nid = f"L{layer_idx}N{neuron_idx}"
                graph.add_node(nid)
                
                vertical_pos = (layer.neurons - 1) / 2 - neuron_idx if layer.neurons > 1 else 0
                positions[nid] = (layer_idx * 2, vertical_pos)
                labels[nid] = nid
        
        for i in range(len(self.network) - 1):
            layer = self.network[i]
            
            if layer.weight_to is None:
                continue
                
            for j in range(layer.neurons):
                for k in range(self.network[i+1].neurons):
                    source = f"L{i}N{j}"
                    target = f"L{i+1}N{k}"
                    
                    try:
                        w_val = layer.weight_to[j, k] if j < layer.weight_to.shape[0] and k < layer.weight_to.shape[1] else 0
                    except IndexError:
                        w_val = 0
                    
                    g_val = None
                    if show_gradients and hasattr(layer, 'weight_grad') and layer.weight_grad is not None:
                        try:
                            if j < layer.weight_grad.shape[0] and k < layer.weight_grad.shape[1]:
                                g_val = layer.weight_grad[j, k]
                        except IndexError:
                            pass
                    
                    graph.add_edge(source, target, weight=w_val, gradient=g_val)
        
        plt.figure(figsize=(14, 10))
        
        nx.draw_networkx_nodes(graph, positions, node_size=700, node_color='lightblue', alpha=0.8)
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=10)
        
        conn = graph.edges(data=True)
        colors = []
        widths = []
        
        for u, v, data in conn:
            w = data['weight']
            colors.append('red' if w < 0 else 'blue')
            widths.append(1 + abs(w))
        
        nx.draw_networkx_edges(graph, positions, edgelist=conn, width=widths, 
                        edge_color=colors, arrows=True, arrowsize=15, alpha=0.6)
        
        if show_weights or show_gradients:
            edge_text = {}
            
            for u, v, data in graph.edges(data=True):
                parts = []
                
                if show_weights:
                    parts.append(f"W: {data['weight']:.2f}")
                
                if show_gradients and data['gradient'] is not None:
                    parts.append(f"G: {data['gradient']:.2f}")
                
                edge_text[(u, v)] = "  ".join(parts)
            
            nx.draw_networkx_edge_labels(
                graph, positions, 
                edge_labels=edge_text, 
                font_size=8,
                font_family='sans-serif',
                font_weight='normal',
                font_color='black',
                bbox=dict(alpha=0.6, pad=2, edgecolor='gray', facecolor='white'),
                label_pos=0.4
            )
        
        plt.title("Neural Network Structure")
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"Network graph visualization saved as {filename}")
        else:
            try:
                plt.show()
            except:
                default_filename = f"network_viz_{show_weights}_{show_gradients}.png"
                plt.savefig(default_filename, dpi=150)
                print(f"Network graph visualization saved as {default_filename}")
        
        plt.close()

    def save(self):
        """
        Saves the model instance using a format (json?)
        """
        pass

layer_i = Layer(sigmoid, [random_uniform, {"seed": 42, "lb": 0, "ub": 0.5}], 2)
layer_1 = Layer(sigmoid, [random_uniform, {"seed": 60, "lb": 0, "ub": 0.5}], 2)
layer_o = Layer(None, [None, {}], 2)
ann = ANN(None, [layer_1], input=layer_i, output=layer_o, error=SSE)
ann.train(l_rate=0.5, epoch=10000)
ann.show()
# print((np.array([1, 2]) + np.array([3, 4])) / 2)

