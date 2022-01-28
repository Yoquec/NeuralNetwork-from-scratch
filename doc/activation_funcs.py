import numpy as np
from dummydata import spiral_data

#NOTES: -----------------------------------------------------
"""
There are various types of activation functions, such as the
"Rectified Linear", the "Unit Step", "Sigmoid" and "SoftMax".

Activation functions
https://youtu.be/gmjzbpSVY1A?t=439
"""
#-------------------------------------------------------------
# X = [[1.0, 2.0, 3.0, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

class Activation_RelU:
    def forward(self, inputs):
        """Implements the ReLu activation"""
        self.output = np.maximum(0, inputs)

# Left in 32:30

if __name__ == "__main__":
    np.random.seed(0)

    """
    SoftMax Example
    """
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
    output = np.zeros((len(inputs)))

    for i in range(len(inputs)):
        val = inputs[i]
        if val > 0:
            output[i] = val
        else:
            output[i] = 0

# NOTE: Can also be done as
    for k in range(len(inputs)):
        output[k] = max(0, inputs[k])

    print(output)
