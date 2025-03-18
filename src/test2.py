from ANN import *
import numpy as np

def test_gradient_visualization():
    layer_i = Layer(relU, [random_uniform, {"seed": 42, "lb": 0, "ub": 0.5}], 2)
    layer_1 = Layer(relU, [random_uniform, {"seed": 60, "lb": 0, "ub": 0.5}], 2)
    layer_o = Layer(None, [None, {}], 1)
    
    ann = ANN(None, [layer_1], input=layer_i, output=layer_o, error=SSE)
    
    ann.network[0].weight_grad = np.random.randn(*ann.network[0].weight_to.shape)
    ann.network[1].weight_grad = np.random.randn(*ann.network[1].weight_to.shape)
    
    print("ANN structure:")
    ann.show()
    
    print("\nTest 1: Visualizing all layers with weight gradients")
    ann.wg_dist_show()
    
    print("\nTest 2: Visualizing layer 0 gradients")
    ann.wg_dist_show([0])
    
    print("\nTest 3: Visualizing layer 1 gradients")
    ann.wg_dist_show([1])
    
    print("\nTest 4: Visualizing layers 0 and 1 gradients together")
    ann.wg_dist_show([0, 1])
    
    print("\nTest 5: Testing with invalid layer")
    ann.wg_dist_show([0, 3])

if __name__ == "__main__":
    test_gradient_visualization()
