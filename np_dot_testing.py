import numpy as np

#########################################
# 4 previous neurons and 1 current case
#########################################
# Example (with 3 previous neurons):
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
# Unique bias
bias = 2

# The output is to add up all the inputs times the weights plus the bias.
output = np.dot(inputs, weights) + bias

# print the output:
print(f"The value of the neuron ouput is {output}")

#########################################
# 4 previous neurons and 3 current case 
#########################################
# We will use the previous input
weights0 = [0.2, 0.8, -0.5, 1.0]
weights1 = [-3.3, 2.8, 5.4, -3.2]
weights2 = [1.6, 3.6, -0.3, -3.6]
# Make the new vector of biases
biases = [1, 2, 3]

weightsmul = [weights0, weights1, weights2]

# Be carefull with the dimensions of the vectors when doing matrix multiplication!!!
outputmul = np.dot(weightsmul, inputs) + biases

print(f"The output of the 3 neurons will be {outputmul}")