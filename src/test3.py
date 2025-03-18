from ANN import *
import numpy as np

def test_network_visualization():
    layer_i = Layer(relU, [random_uniform, {"seed": 42, "lb": 0, "ub": 0.5}], 2)
    layer_1 = Layer(relU, [random_uniform, {"seed": 60, "lb": 0, "ub": 0.5}], 2)
    layer_o = Layer(None, [None, {}], 1)
    
    ann = ANN(None, [layer_1], input=layer_i, output=layer_o, error=SSE)
    
    ann.network[0].weight_grad = np.random.randn(*ann.network[0].weight_to.shape)
    ann.network[1].weight_grad = np.random.randn(*ann.network[1].weight_to.shape)
    
    print("Test 1: Visualizing network with weights only")
    ann.visualize_network(show_weights=True, show_gradients=False, filename="network_weights_only.png")
    
    print("Test 2: Visualizing network with weights and gradients")
    ann.visualize_network(show_weights=True, show_gradients=True, filename="network_weights_gradients.png")
    
    large_layer_i = Layer(relU, [random_uniform, {"seed": 42, "lb": 0, "ub": 0.5}], 4)
    large_layer_1 = Layer(relU, [random_uniform, {"seed": 60, "lb": 0, "ub": 0.5}], 3)
    large_layer_2 = Layer(relU, [random_uniform, {"seed": 70, "lb": 0, "ub": 0.5}], 3)
    large_layer_o = Layer(None, [None, {}], 2)
    
    try:
        large_ann = ANN(None, [large_layer_1, large_layer_2], input=large_layer_i, output=large_layer_o, error=SSE)
        
        print("Test 3: Visualizing larger network")
        large_ann.visualize_network(filename="larger_network.png")
    except Exception as e:
        print(f"Test 3 failed due to: {e}")

if __name__ == "__main__":
    test_network_visualization()
