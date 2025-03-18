import matplotlib
matplotlib.use('Agg')
# ...existing code...

# After creating your ReLU network
relu_ann = ANN(...)
relu_ann.train(l_rate=0.1, epoch=1)  # At least one epoch to initialize
relu_ann.show()

# ...existing code...

# Save plots to files instead of showing them
plt.savefig("network_visualization.png")
