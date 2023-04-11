# Referring to mathematical formula for backward propagation see script
# vanishing_gradients
"""
In the backpropagation the output of the NN is fed backwards through the chain-rule to the input neurons.
Through the chain-rule each layer value and activation function is split up and multiplied.
That's where the vanishing gradient can occur if the gradient of the weights or the activation function is not near to 1.
Since multiple layers are stacked in series, these multiply to nearly 0 (vanishing gradients) for gradients < 1,
assuming there are enough layers in series.
"""
# dying_ReLUs
"""
A rectified linear unit composes of two parts: the linear activation (usually for x > 0) 
and the constant zero activation (usually for x <= 0).
If a ReLU neuron has a negative input, the output is therefore set to 0.
In the backpropagation the resulting gradient for this neuron will therefore always be 0 too.
This also means that the neuron will be 0 in the next iteration because the update in the gradient
descent method is multiplied by the backward gradient.
This is then called a DeadRelu, since the activation function does not change and this neuron stays constant 0 (dead).
"""