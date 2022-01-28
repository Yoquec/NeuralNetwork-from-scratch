################################
# Neurons
################################
# All neurons have unique inputs (the outputs of the previous neurons), weights and a bias

# Example (with 3 previous neurons):
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
# Unique bias
bias = 2

# The output is to add up all the inputs times the weights plus the bias.
# (can be done faster with np.dot)
output = inputs[0] * weights[0] +\
         inputs[1] * weights[1] +\
         inputs[2] * weights[2] +\
         inputs[3] * weights[3] + bias

# print the output:
print(f"The value of the neuron ouput is {output}")


#####################################
# Modelling 3 neurons with 4 inputs
#####################################
# This is going to be a very simple explanation, no loops. no algegra
# just raw python declarations. Later numpy and more complex structures will be used

#-------------------- 0
inputs0 = [1.0, 2.0, 3.0, 2.5]
weights0 = [0.2, 0.8, -0.5, 1.0]
bias0 = 3

# Compute the output:
output0 = inputs0[0] * weights0[0] +\
         inputs0[1] * weights0[1] +\
         inputs0[2] * weights0[2] +\
         inputs0[3] * weights0[3] + bias0
#-------------------- 1
inputs1 = [3.0, 1.0, 1.5, 1.4]
weights1 = [-3.3, 2.8, 5.4, -3.2]
bias1 = 1

# Compute the output:
output1 = inputs1[0] * weights1[0] +\
         inputs1[1] * weights1[1] +\
         inputs1[2] * weights1[2] +\
         inputs1[3] * weights1[3] + bias1
#-------------------- 2
inputs2 = [-4.0, 2.0, 1.5, -1.5]
weights2 = [1.6, 3.6, -0.3, -3.6]
bias2 = 4

# Compute the output:
output2 = inputs2[0] * weights2[0] +\
         inputs2[1] * weights2[1] +\
         inputs2[2] * weights2[2] +\
         inputs2[3] * weights2[3] + bias2
#-------------------- 

# Make the output vector
output = [output0, output1,output2]

print(f"The value of the three neurons' ouput is {output}")