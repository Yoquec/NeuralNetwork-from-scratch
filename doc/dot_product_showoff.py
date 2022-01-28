# Converting of sample of inputs into batches of inputs
# Modelling multiple layers of neurons.
# Calculating batches can be done in parallel, very optimized for gpu's
# Also batches help for generalization of input data.

import numpy as np

#Inizialize inputs, weights and biases for the 3 neurons
inputs = np.array([[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]])

biases = [2, 3, 0.5]

#Compute the output of the three neurons with matrix math
l1_output = np.dot(inputs, weights.T) + biases

print (l1_output)

# Make a second layer after layer 1 (takes layer 1 as input)
weights2 = np.array([
        [0.1, -0.14, 0.5],
        [-0.5, -0.12, -0.33],
        [-0.44, 0.73, -0.13]
        ])

biases2 = [-1, 2, -0.5]

# Compute the output of  the second layer
l2_output = np.dot(l1_output, weights2.T) + biases2

print(f"\nThe output of the second layer is:\n{l2_output}")
