import numpy as np
from typing import List

#NOTES: -----------------------------------------------------
"""
INITIALIZING WEIGHTS:
Weights usually are inizialized as random values between -1 and 1
and avoid non-normalized values because they can "scale" the 
"importance" of each weigh.

In this case we will initialize weigths with values between -.1 and .1


INITIALIZING BIASES
We can initialize biases with a small value. We should avoid 0 bias to avoid the
propagation of zeros across all the network.
"""
#-------------------------------------------------------------


class Layer:
    """Class to represent neuron layers and batches"""
    def __init__(self, n_in, n_neurons):
        # Create random weigths (use 0.10 to scale between 0 and 1)
        self.weigths = 0.10 * np.random.randn(n_in, n_neurons) #Will have correct dimensions (no need for transposing in the forward pass)

        # Zero take a tuple with the dimensions of the 
        self._bias = np.zeros((1, n_neurons))

    def forward_pass(self, inputs: List[List[float]]) -> None:
        """Function to calculate the output of the layer of neurons.
           The function will not return any value, but instead set the property self.output."""
        self.output = np.dot(inputs, self.weigths) + self._bias


if __name__ == "__main__":
    np.random.seed(0)

    # Inizialize the input matrix
    X = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

    # Create the layers
    layer1 = Layer(4, 5)
    layer2 = Layer(5, 2)

    # Pass data through them
    layer1.forward_pass(X)
    layer2.forward_pass(layer1.output)

    # Print the results
    print(f"The output of the first neuron is\n{layer1.output}")
    print(f"\nThe output of the second neuron is\n{layer2.output}")
