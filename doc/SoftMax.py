"""
SOFTMAX ACTIVATION
It is used to be able to approximate the output values of neurons
to probability distributions. If we were to use softmax for example,
negative values would be clipped to 0 and no proper probability distribution
could be fit, because no matter the magnitude, different negatives will give
all 0.

SoftMax uses exponentiation to rectify the values (y = e^x)
"""
import math

import numpy 

# Implement exponentiation
layer_outputs = [4.8, 1.21, 2.385]

exp_vals = []

for out in layer_outputs:
    exp_vals.append(pow(math.e, out))

print(f"Non normalized values are {exp_vals}\n")

#Implement normalization of outputs
norm_factor = sum(exp_vals)

norm_values = []

for val in exp_vals:
    norm_values.append(val/norm_factor)

print(f"Normalized values are {norm_values}\n")
print(sum(norm_values))



#####################################
# Now using numpy
#####################################
# 13:14
