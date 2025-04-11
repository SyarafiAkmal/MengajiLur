import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  
import networkx as nx
import pickle
import time

#=================PENTINGGGGGGGG===============

# UBAH DULU KE NUMPY ARRAY DATASETNYA

#===============================================

# activation functions

def linear(input, der=False):
    if not der:
        return input
    return np.ones_like(input)

def relU(input, der=False):
    if not der:
        return np.maximum(0, input)
    return np.where(input > 0, 1, 0)

def eLU(input, der=False):
    if not der:
        return np.where(input > 0, input, np.exp(input) - 1)
    return np.where(input > 0, 1, np.exp(input))

def sigmoid(input, der=False):
    inp_safe = np.clip(input, -500, 500)
    sig = 1 / (1 + np.exp(-inp_safe))
    
    if not der:
        return sig
    return sig * (1 - sig)

def h_tan(input, der=False):
    if not der:
        return np.tanh(input)
    return 1 - np.tanh(input)**2

def softmax(input, der=False):
    if not der:
        shifted = input - np.max(input, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        s = softmax(input)
        return s * (1 - s)

def step_function(input, der=False):
    if der:
        return 0
    return np.where(input >= 0, 1, 0)

# error loss functions

def SSE(target, pred, der=False):
    if not der:
        return 0.5 * np.sum((target - pred) ** 2)
    return (pred - target)  

def MSE(target, pred, der=False):
    if not der:
        return np.mean(np.square(target - pred))
    return (pred - target) * 2 

def BCE(target, pred, der=False):
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1 - epsilon)
    
    if not der:
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    else:
        target = np.asarray(target, dtype=float)
        return ((1 - target) / (1 - pred) - target / pred)

def CCE(target, pred, der=False):
    # Well, if the target is exactly 0
    # The log will be undefined, I guess
    # So, Let's clip it
    # Again, CMIIW
    epsilon = 1e-15
    pred = np.clip(pred, epsilon, 1.0)
    
    if not der:
        if len(pred.shape) == 2 and (len(target.shape) == 1 or target.shape[1] == 1):
            one_hot_target = np.zeros_like(pred)
            for i in range(len(target)):
                idx = int(target[i]) if len(target.shape) == 1 else int(target[i, 0])
                one_hot_target[i, idx] = 1
            target = one_hot_target
            
        return -np.mean(np.sum(target * np.log(pred), axis=1))
    else:
        if len(pred.shape) == 2 and (len(target.shape) == 1 or target.shape[1] == 1):
            one_hot_target = np.zeros_like(pred)
            for i in range(len(target)):
                idx = int(target[i]) if len(target.shape) == 1 else int(target[i, 0])
                one_hot_target[i, idx] = 1
            target = one_hot_target
        return (pred - target)

# weight init functions

def zero_init(fr, to, params=None):
    return np.zeros((fr+1, to))

def random_uniform(fr, to, params=None):
    if params is None:
        params = {"seed": None, "lb": -0.5, "ub": 0.5}
    
    if "seed" in params and params["seed"] is not None:
        np.random.seed(params["seed"])
    # if "seed" in params and params["seed"] == 42:
    #     return np.array([[0.35, 0.35], [0.15, 0.25], [0.20, 0.30]])
    
    # if "seed" in params and params["seed"] == 60:
    #     return np.array([[0.6, 0.6], [0.40, 0.50], [0.45, 0.55]])
    
    lb = params.get("lb", -0.5)
    ub = params.get("ub", 0.5)
    
    return np.random.uniform(lb, ub, (fr+1, to))

def random_normal(fr, to, params=None):
    if params is None:
        params = {"seed": None, "mean": 0, "var": 0.1}
        
    if "seed" in params and params["seed"] is not None:
        np.random.seed(params["seed"])
    
    mean = params.get("mean", 0)
    var = params.get("var", 0.1)
    
    return np.random.normal(mean, np.sqrt(var), (fr+1, to))

def xavier_init(fr, to, params=None):
    if params is None:
        params = {"seed": None}
    
    if "seed" in params and params["seed"] is not None:
        np.random.seed(params["seed"])
    
    limit = np.sqrt(6 / (fr + to))
    return np.random.uniform(-limit, limit, (fr+1, to))

def he_init(fr, to, params=None):
    if params is None:
        params = {"seed": None}
    
    if "seed" in params and params["seed"] is not None:
        np.random.seed(params["seed"])
    
    std = np.sqrt(2 / fr)
    return np.random.normal(0, std, (fr+1, to))

# classes

class Layer:
    def __init__(self, a_func, w_init, neurons):
        self.neurons = neurons
        self.a_func = a_func
        self.w_init = w_init
        self.input = None
        self.net = None
        self.output = None
        self.error = None
        self.weight_to = None
        self.weight_grad = None 

class ANN:
    def __init__(self, data=None, config=None, input=None, output=None, error=None, load_path=None, reg=None, lambda_reg=0.0):
        self.data = data
        self.err_func = error
        self.history = {"train_loss": [], "val_loss": []}
        self.reg = reg
        self.lambda_reg = lambda_reg
        
        if load_path:
            self.load(load_path)
            return
            
        if config is None:
            config = []
            
        self.network = []
        if input:
            self.network.append(input)
        self.network.extend(config)
        if output:
            self.network.append(output)
        
        for i in range(len(self.network)-1):
            if self.network[i].w_init[0] is not None:
                self.network[i].weight_to = self.network[i].w_init[0](
                    self.network[i].neurons, 
                    self.network[i+1].neurons, 
                    self.network[i].w_init[1]
                )
                self.network[i].weight_grad = np.zeros_like(self.network[i].weight_to)


    def forward_propagation(self, X):
        batch_size = X.shape[0]
        
        X_with_bias = np.hstack((np.ones((batch_size, 1)), X))
        
        self.network[0].input = X_with_bias
        
        # Process through each layer
        for i in range(len(self.network)-1):
            print(self.network[i].input.shape)
            print(self.network[i].weight_to.shape)
            self.network[i+1].net = self.network[i].input @ self.network[i].weight_to
            
            if i == len(self.network)-2: 
                if self.network[i+1].a_func is not None:
                    self.network[i+1].output = self.network[i+1].a_func(self.network[i+1].net)
                else:
                    self.network[i+1].output = self.network[i+1].net
                self.network[i+1].input = self.network[i+1].output
            else: 
                self.network[i+1].output = self.network[i+1].a_func(self.network[i+1].net)
                self.network[i+1].input = np.hstack((np.ones((batch_size, 1)), self.network[i+1].output))

        return self.network[-1].output
    
    def backward_propagation(self, X, y):
        batch_size = X.shape[0]
        
        output_layer = self.network[-1]

        # print(output_layer)
        
        if output_layer.a_func == sigmoid and (self.err_func == MSE or self.err_func == SSE):
            # Sigmoid and MSE loss 
            output_error = (output_layer.output - y) 
            if self.err_func == MSE:
                output_error = output_error * 2  
            activation_derivative = output_layer.output * (1 - output_layer.output)  
            output_layer.error = output_error * activation_derivative
        elif output_layer.a_func == softmax and self.err_func == CCE:
            # Softmax + CCE
            if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
                y_one_hot = np.zeros((batch_size, output_layer.output.shape[1]))
                for i in range(batch_size):
                    idx = int(y[i]) if len(y.shape) == 1 else int(y[i, 0])
                    y_one_hot[i, idx] = 1
                y = y_one_hot
            output_layer.error = (output_layer.output - y)
        else:
            # General case 
            loss_derivative = self.err_func(y, output_layer.output, der=True)
            
            if output_layer.a_func is not None:
                activation_derivative = output_layer.a_func(output_layer.net, der=True)
                output_layer.error = loss_derivative * activation_derivative
            else:
                output_layer.error = loss_derivative
        
        # Backpropagate error 
        for i in range(len(self.network)-2, 0, -1):
            current_layer = self.network[i]
            next_layer = self.network[i+1]
            
            error_contrib = next_layer.error @ self.network[i].weight_to[1:].T
            
            activation_derivative = current_layer.a_func(current_layer.net, der=True)
            current_layer.error = error_contrib * activation_derivative
        
        # Weight gradients 
        for i in range(len(self.network)-1):
            # self.network[i].weight_grad = self.network[i].input.T @ self.network[i+1].error

            grad = self.network[i].input.T @ self.network[i+1].error

            if self.reg == 'l1':
                reg_term = self.lambda_reg * np.sign(self.network[i].weight_to)
                reg_term[0, :] = 0
                grad += reg_term
            elif self.reg == 'l2':
                reg_term = self.lambda_reg * self.network[i].weight_to
                reg_term[0, :] = 0
                grad += reg_term

            self.network[i].weight_grad = grad

    def update_weights(self, learning_rate):
        batch_size = self.network[0].input.shape[0]  
        for i in range(len(self.network)-1):
            self.network[i].weight_to -= learning_rate * (self.network[i].weight_grad / batch_size)

    def train(self, X_train=None, y_train=None, batch_size=32, l_rate=0.01, epoch=10, X_val=None, y_val=None, verb=0):
        self.history = {"train_loss": [], "val_loss": []}

        # print(self.history)
        
        if X_train is None or y_train is None:
            print("No training data provided")
            return self.history
        
        if hasattr(X_train, 'to_numpy'):
            X_train = X_train.to_numpy()
        if hasattr(y_train, 'to_numpy'):
            y_train = y_train.to_numpy()

        if X_val is not None and hasattr(X_val, 'to_numpy'):
            X_val = X_val.to_numpy()
        if y_val is not None and hasattr(y_val, 'to_numpy'):
            y_val = y_val.to_numpy()
        
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # print(l_rate)
        
        for e in range(epoch):
            epoch_start = time.time()
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            batch_losses = []
            
            # Process batches
            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min((b + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = self.forward_propagation(X_batch)
                
                # Loss
                batch_loss = self.err_func(y_batch, output)
                if self.reg in ['l1', 'l2']:
                    reg_loss = 0
                    for layer in self.network[:-1]:
                        if layer.weight_to is not None:
                            weights_wo_bias = layer.weight_to[1:, :]
                            if self.reg == 'l1':
                                reg_loss += np.sum(np.abs(weights_wo_bias))
                            elif self.reg == 'l2':
                                reg_loss += np.sum(np.square(weights_wo_bias))
                    batch_loss += self.lambda_reg * reg_loss
                batch_losses.append(batch_loss)
                
                # Backward pass
                self.backward_propagation(X_batch, y_batch)
                
                # Update weights
                self.update_weights(l_rate)
            
            # Average loss for epoch
            epoch_loss = np.mean(batch_losses)
            self.history["train_loss"].append(epoch_loss)
            
            if X_val is not None and y_val is not None:
                val_output = self.forward_propagation(X_val)
                val_loss = self.err_func(y_val, val_output)
                self.history["val_loss"].append(val_loss)
            
            # Verbose
            if verb > 0:
                epoch_end = time.time()
                epoch_time = epoch_end - epoch_start
                
                if e % (max(1, epoch // 10)) == 0 or e == epoch-1:
                    if X_val is not None and y_val is not None:
                        print(f"Epoch {e+1}/{epoch} - loss: {epoch_loss:.6f} - val_loss: {val_loss:.6f} - time: {epoch_time:.2f}s")
                    else:
                        print(f"Epoch {e+1}/{epoch} - loss: {epoch_loss:.6f} - time: {epoch_time:.2f}s")
        
        return self.history
    
    def predict(self, X):
        return self.forward_propagation(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        return self.err_func(y, predictions)

    def show(self):
        print("ANN structure:")
        for index, layer in enumerate(self.network):
            if index == 0:
                print("input layer")
            elif index == len(self.network) - 1:
                print("output layer")
            else:
                print(f"hidden layer {index}")
            print(f"{layer.neurons} neurons")
            
            if hasattr(layer, 'weight_to') and layer.weight_to is not None:
                print("Weight shape:", layer.weight_to.shape)
                if layer.weight_to.size < 20:
                    print("Weights:\n", layer.weight_to)
            
            if layer.a_func is not None:
                func_name = layer.a_func.__name__
                print(f"Activation function: {func_name}")
            
            if layer.error is not None:
                print("Error shape:", layer.error.shape)
            print("")
        
        print(f"Loss function: {self.err_func.__name__}")
        if len(self.history['train_loss']) > 0:
            print(f"Final training loss: {self.history['train_loss'][-1]:.6f}")
            if len(self.history.get('val_loss', [])) > 0:
                print(f"Final validation loss: {self.history['val_loss'][-1]:.6f}")
    
    def error_translate(self, a_func, x, y):
        # x * (1 - x): derivative of the a_func
        # (y - x): derivative of the loss func 
        return (a_func(x, True) * self.err_func(y, x, True))
    
    def weight_update(self, weight, l_rate, error, input):
        return weight - ((l_rate * error) * input)
    
    def save(self, filepath):
        model_data = {
            'network': self.network,
            'err_func_name': self.err_func.__name__,
            'history': self.history
        }

        # print("=================MODEL DATA=================")
        # print(model_data)
        # print("===========================================")
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.network = model_data['network']
        
        error_funcs = {
            'SSE': SSE,
            'MSE': MSE,
            'BCE': BCE,
            'CCE': CCE
        }
        self.err_func = error_funcs.get(model_data['err_func_name'])
        
        self.history = model_data.get('history', {"train_loss": [], "val_loss": []})
        
        print(f"Model loaded from {filepath}")
    
    def visualize_network(self, show_weights=True, show_gradients=False, filename=None, figsize=(14, 10), show_bias=True, weight_table=False, weight_table_filename=None):
        graph = nx.DiGraph()
        positions = {}
        labels = {}
        node_colors = []
        node_sizes = []
        
        for layer_idx, layer in enumerate(self.network):
            layer_name = "Input Layer" if layer_idx == 0 else \
                         "Output Layer" if layer_idx == len(self.network) - 1 else \
                         f"Hidden Layer {layer_idx}"
            if layer.a_func is not None:
                act_name = layer.a_func.__name__
                layer_name += f"\n({act_name})"
            
            graph.add_node(f"L{layer_idx}", layer=True)
            positions[f"L{layer_idx}"] = (layer_idx * 4, 2)
            labels[f"L{layer_idx}"] = layer_name
            node_colors.append('lightgreen')
            node_sizes.append(1200)
            
            neuron_count = layer.neurons
            if show_bias and layer_idx < len(self.network) - 1:
                neuron_count += 1
                
            for neuron_idx in range(neuron_count):
                is_bias = (neuron_idx == 0 and show_bias and layer_idx < len(self.network) - 1)
                nid = f"L{layer_idx}N{neuron_idx}"
                graph.add_node(nid, bias=is_bias)
                
                vertical_pos = -2 - (neuron_idx * 1.5)
                positions[nid] = (layer_idx * 4, vertical_pos)
                
                if is_bias:
                    labels[nid] = "Bias"
                    node_colors.append('orange')
                    node_sizes.append(600)
                else:
                    neur_num = neuron_idx if not show_bias else neuron_idx - 1
                    labels[nid] = f"N{neur_num}"
                    node_colors.append('lightblue')
                    node_sizes.append(600)
        
        weight_table_data = []
        
        for i in range(len(self.network) - 1):
            layer = self.network[i]
            next_layer = self.network[i+1]
            
            if layer.weight_to is None:
                continue
            
            source_neurons = layer.neurons + (1 if show_bias else 0)
            target_neurons = next_layer.neurons
            
            for j in range(source_neurons):
                for k in range(target_neurons):
                    source = f"L{i}N{j}"
                    target_idx = k
                    if show_bias and i+1 < len(self.network) - 1:
                        target_idx += 1
                    target = f"L{i+1}N{target_idx}"
                    
                    if source not in positions or target not in positions:
                        continue
                    
                    w_val = layer.weight_to[j, k]
                    
                    g_val = None
                    if show_gradients and layer.weight_grad is not None:
                        g_val = layer.weight_grad[j, k]
                    
                    graph.add_edge(source, target, weight=w_val, gradient=g_val)
                    
                    if weight_table:
                        source_label = "Bias" if (j == 0 and show_bias) else f"N{j if not show_bias else j-1}"
                        target_label = f"N{k}"
                        source_layer_type = "Input" if i == 0 else "Hidden" if i < len(self.network) - 1 else "Output"
                        target_layer_type = "Hidden" if i+1 < len(self.network) - 1 else "Output"
                        
                        weight_table_data.append({
                            'source': f"{source_layer_type} L{i} {source_label}",
                            'target': f"{target_layer_type} L{i+1} {target_label}",
                            'weight': w_val,
                            'gradient': g_val if g_val is not None else "N/A"
                        })
        
        plt.figure(figsize=figsize)
        
        nx.draw_networkx_nodes(graph, positions, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_labels(graph, positions, labels=labels, font_size=10)
        
        conn = list(graph.edges(data=True))
        colors = []
        widths = []
        
        for u, v, data in conn:
            w = data['weight']
            colors.append('red' if w < 0 else 'blue')
            widths.append(0.5 + min(3, abs(w)))
        
        nx.draw_networkx_edges(graph, positions, edgelist=conn, width=widths, 
                              edge_color=colors, arrows=True, arrowsize=15, alpha=0.6)
        
        if (show_weights or show_gradients) and not weight_table:
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
        
        plt.title("Neural Network Architecture")
        plt.axis('off')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"Network visualization saved to {filename}")
        else:
            plt.show()
        
    def w_dist_show(self, layers=None, figsize=(12, 5), filename=None):
        if layers is None:
            layers = []
            for i in range(len(self.network)):
                if self.network[i].weight_to is not None:
                    layers.append(i)
        
        good_layers = []
        for idx in layers:
            if idx >= 0 and idx < len(self.network):
                if self.network[idx].weight_to is not None:
                    good_layers.append(idx)
        
        if not good_layers:
            print("No layers to visualize.")
            return
        
        width = min(6 * len(good_layers), figsize[0])
        fig, axes = plt.subplots(1, len(good_layers), figsize=(width, figsize[1]))
        
        if len(good_layers) == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(good_layers):
            layer = self.network[layer_idx]
            w = layer.weight_to.flatten()
            
            w_mean = np.mean(w)
            w_std = np.std(w)
            w_min = np.min(w)
            w_max = np.max(w)
            
            axes[i].hist(w, bins=30, color='blue', alpha=0.7)
            axes[i].axvline(w_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {w_mean:.4f}')
            
            stats_text = f"Mean: {w_mean:.4f}\nStd: {w_std:.4f}\nMin: {w_min:.4f}\nMax: {w_max:.4f}"
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
            layer_type = "Input" if layer_idx == 0 else "Hidden" if layer_idx < len(self.network) - 1 else "Output"
            next_layer_idx = layer_idx + 1
            axes[i].set_title(f'Layer {layer_idx} → {next_layer_idx} ({layer_type}→{layer_type})')
            axes[i].set_xlabel('Weight Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"Weight visualization saved to {filename}")
        else:
            plt.show()
        
    def wg_dist_show(self, layers=None, figsize=(12, 5), filename=None):

        # There is something wrong with the seeding, must revisit later
        
        if layers is None:
            layers = []
            for i in range(len(self.network)):
                if self.network[i].weight_grad is not None:
                    layers.append(i)
        
        valid_layer_list = []
        for l_idx in layers:
            if l_idx >= 0 and l_idx < len(self.network) and self.network[l_idx].weight_grad is not None:
                valid_layer_list.append(l_idx)
        
        if not valid_layer_list:
            print("No gradients to visualize.")
            return
        
        width = min(6 * len(valid_layer_list), figsize[0])
        fig, ax_array = plt.subplots(1, len(valid_layer_list), figsize=(width, figsize[1]))
        
        if len(valid_layer_list) == 1:
            ax_array = [ax_array]
        
        i = 0
        for l_idx in valid_layer_list:
            layer = self.network[l_idx]
            grads = layer.weight_grad.flatten()
            
            g_mean = np.mean(grads)
            g_std = np.std(grads) 
            g_min = np.min(grads)
            g_max = np.max(grads)
            
            ax_array[i].hist(grads, bins=30, color='red', alpha=0.7)
            ax_array[i].axvline(g_mean, color='blue', linestyle='dashed', linewidth=1, label=f'Mean: {g_mean:.6f}')
            
            stats_text = f"Mean: {g_mean:.6f}\nStd: {g_std:.6f}\nMin: {g_min:.6f}\nMax: {g_max:.6f}"
            ax_array[i].text(0.02, 0.98, stats_text, transform=ax_array[i].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
            layer_type = "Input" if l_idx == 0 else "Hidden" if l_idx < len(self.network) - 1 else "Output"
            next_layer_idx = l_idx + 1
            ax_array[i].set_title(f'Layer {l_idx} → {next_layer_idx} Gradients ({layer_type}→{layer_type})')
            ax_array[i].set_xlabel('Gradient Value')
            ax_array[i].set_ylabel('Frequency')
            ax_array[i].grid(True, linestyle='--', alpha=0.7)
            ax_array[i].legend()
            i += 1
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"Gradient visualization saved to {filename}")
        else:
            plt.show()
    
    def plot_training_history(self, figsize=(10, 6), filename=None):
        plt.figure(figsize=figsize)
        
        if len(self.history["train_loss"]) > 0:
            plt.plot(self.history["train_loss"], 'b-', label='Training Loss')
        
        if "val_loss" in self.history and self.history["val_loss"]:
            plt.plot(self.history["val_loss"], 'r-', label='Validation Loss')
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"History plot saved to {filename}")
        else:
            plt.show()

if __name__ == "__main__":
    X = np.array([[0.05, 0.1]])
    y = np.array([[0.01, 0.99]])
    
    layer_i = Layer(sigmoid, [xavier_init, {"seed": 42, "lb": -0.5, "ub": 0.5}], 2)
    layer_1 = Layer(sigmoid, [xavier_init, {"seed": 60, "lb": -0.5, "ub": 0.5}], 2)
    layer_o = Layer(sigmoid, [None, {}], 2)
    
    ann = ANN(None, [layer_1, layer_1, layer_1], input=layer_i, output=layer_o, error=MSE, reg="l2", lambda_reg=0.01)
    ann.train(X, y, batch_size=4, l_rate=0.5, epoch=2, verb=1)
    
    
    predictions = ann.predict(X)
    ann.show()
    print("Predictions:")
    print(predictions)