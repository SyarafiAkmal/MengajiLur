from ANN import *

def test_weight_visualization():
    layer_i = Layer(relU, [random_uniform, {"seed": 42, "lb": 0, "ub": 0.5}], 2)
    layer_1 = Layer(relU, [random_uniform, {"seed": 60, "lb": 0, "ub": 0.5}], 2)
    layer_o = Layer(None, [None, {}], 1)
    
    ann = ANN(None, [layer_1], input=layer_i, output=layer_o, error=SSE)
    
    print("ANN structure:")
    ann.show()
    
    print("\nTest 1: Visualizing all layers with weights")
    input("Enter...")
    ann.w_dist_show()
    
    print("\nTest 2: Visualizing layer 0")
    ann.w_dist_show([0])
    
    print("\nTest 3: Visualizing layer 1")
    ann.w_dist_show([1])
    
    print("\nTest 4: Visualizing layers 0 and 1 together")
    ann.w_dist_show([0, 1])
    
    print("\nTest 5: Testing with invalid layer")
    ann.w_dist_show([0, 3])

if __name__ == "__main__":
    test_weight_visualization()
