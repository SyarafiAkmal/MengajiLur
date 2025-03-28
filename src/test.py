import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from ANN import *

def load_mnist(sample_size=5000):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    X = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
    y = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
    y = y.astype(int)
    
    if sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X, y = X[idx], y[idx]
    
    X = X / 255.0
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def to_one_hot(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def test_activations():
    X_train, X_test, y_train, y_test = load_mnist(1000)
    y_train_hot = to_one_hot(y_train)
    
    acts = {"ReLU": relU, "Sigmoid": sigmoid, "Tanh": h_tan, "Linear": linear}
    results = {}
    
    for name, act in acts.items():
        inp = Layer(act, [xavier_init, {"seed": 42}], 784)
        hid = Layer(act, [xavier_init, {"seed": 43}], 64)
        out = Layer(softmax, [xavier_init, {"seed": 44}], 10)
        
        model = ANN(None, [hid], input=inp, output=out, error=CCE)
        
        start = time.time()
        model.train(X_train, y_train_hot, batch_size=32, l_rate=0.01, epoch=3, verb=0)
        train_time = time.time() - start
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, np.argmax(preds, axis=1))
        
        results[name] = {'accuracy': acc, 'time': train_time}
        print(f"{name}: Acc={acc:.4f}, Time={train_time:.2f}s")
    
    return results

def test_batches():
    X_train, X_test, y_train, y_test = load_mnist(500)
    y_train_hot = to_one_hot(y_train)
    
    sizes = [1, 16, 64, 128]
    accs, times = [], []
    
    for bs in sizes:
        inp = Layer(relU, [xavier_init, {"seed": 42}], 784)
        hid = Layer(relU, [xavier_init, {"seed": 43}], 64)
        out = Layer(softmax, [xavier_init, {"seed": 44}], 10)
        
        model = ANN(None, [hid], input=inp, output=out, error=CCE)
        
        start = time.time()
        model.train(X_train, y_train_hot, batch_size=bs, l_rate=0.01, epoch=2, verb=0)
        t = time.time() - start
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, np.argmax(preds, axis=1))
        
        accs.append(acc)
        times.append(t)
        print(f"Batch {bs}: Acc={acc:.4f}, Time={t:.2f}s")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar([str(b) for b in sizes], accs)
    plt.title("Accuracy vs Batch Size")
    
    plt.subplot(1, 2, 2)
    plt.bar([str(b) for b in sizes], times)
    plt.title("Training Time vs Batch Size")
    
    plt.tight_layout()
    plt.savefig("batch_comparison.png")
    
    return sizes, accs, times

def test_viz():
    inp = Layer(sigmoid, [xavier_init, {"seed": 42}], 2)
    hid = Layer(sigmoid, [xavier_init, {"seed": 43}], 3)
    out = Layer(sigmoid, [xavier_init, {"seed": 44}], 1)
    
    model = ANN(None, [hid], input=inp, output=out, error=MSE)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    model.train(X, y, batch_size=4, l_rate=0.5, epoch=100, verb=0)
    
    model.visualize_network(filename="network_viz.png")
    model.w_dist_show(filename="weight_dist.png")
    model.wg_dist_show(filename="weight_grad_dist.png")
    model.plot_training_history(filename="training_hist.png")

def test_save_load():
    X_train, X_test, y_train, y_test = load_mnist(1000)
    y_train_hot = to_one_hot(y_train)
    
    inp = Layer(relU, [xavier_init, {"seed": 42}], 784)
    hid = Layer(relU, [xavier_init, {"seed": 43}], 64)
    out = Layer(softmax, [xavier_init, {"seed": 44}], 10)
    
    model = ANN(None, [hid], input=inp, output=out, error=CCE)
    model.train(X_train, y_train_hot, batch_size=32, l_rate=0.01, epoch=2, verb=0)
    
    preds_before = model.predict(X_test[:5])
    model.save("test_model.pkl")
    
    new_model = ANN(load_path="test_model.pkl")
    preds_after = new_model.predict(X_test[:5])
    
    match = np.allclose(preds_before, preds_after)
    print(f"Save/load test: {'Pass' if match else 'Fail'}")
    
    return match

def test_init_methods():
    X_train, X_test, y_train, y_test = load_mnist(1000)
    y_train_hot = to_one_hot(y_train)
    
    inits = {
        "Zero": zero_init,
        "Uniform": random_uniform,
        "Normal": random_normal,
        "Xavier": xavier_init,
        "He": he_init
    }
    
    for name, init in inits.items():
        inp = Layer(relU, [init, {"seed": 42}], 784)
        hid = Layer(relU, [init, {"seed": 43}], 64)
        out = Layer(softmax, [init, {"seed": 44}], 10)
        
        model = ANN(None, [hid], input=inp, output=out, error=CCE)
        model.train(X_train, y_train_hot, batch_size=32, l_rate=0.01, epoch=2, verb=0)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, np.argmax(preds, axis=1))
        
        print(f"{name} init: {acc:.4f}")

def test_losses():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_bin = np.array([[0], [1], [1], [0]])
    
    for loss_name, loss_fn in [("MSE", MSE), ("BCE", BCE)]:
        inp = Layer(sigmoid, [xavier_init, {"seed": 42}], 2)
        hid = Layer(sigmoid, [xavier_init, {"seed": 43}], 4)
        out = Layer(sigmoid, [xavier_init, {"seed": 44}], 1)
        
        model = ANN(None, [hid], input=inp, output=out, error=loss_fn)
        hist = model.train(X, y_bin, batch_size=4, l_rate=0.5, epoch=100, verb=0)
        
        preds = model.predict(X)
        print(f"{loss_name}: loss={hist['train_loss'][-1]:.4f}, preds={preds.flatten().round(2)}")

def main():
    X_train, X_test, y_train, y_test = load_mnist()
    y_hot = to_one_hot(y_train)
    
    inp = Layer(relU, [random_uniform, {"seed": 42}], 784)
    hid = Layer(relU, [random_uniform, {"seed": 43}], 64)
    hid2 = Layer(relU, [random_uniform, {"seed": 43}], 64)
    out = Layer(softmax, [random_uniform, {"seed": 44}], 10)
    
    model = ANN(None, [hid, hid2], input=inp, output=out, error=CCE)
    model.train(X_train, y_hot, batch_size=32, l_rate=0.01, epoch=5, verb=1)
    
    preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, np.argmax(preds, axis=1))
    print(f"Test accuracy: {test_acc:.4f}")
    
    mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='sgd', 
                      max_iter=5, random_state=42, learning_rate_init=0.01, batch_size=32)
    mlp.fit(X_train, y_train)
    mlp_acc = mlp.score(X_test, y_test)
    print(f"sklearn accuracy: {mlp_acc:.4f}")
    
    # test_viz()
    # test_activations()
    test_batches()
    # test_save_load()
    # test_init_methods()
    # test_losses()

if __name__ == "__main__":
    main()