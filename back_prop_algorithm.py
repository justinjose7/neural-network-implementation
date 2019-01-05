"""Back-Prop-Learning Algorithm"""
from helper_functions import (
    sigmoid, derivative_of_sigmoid
)

def back_prop_learning(examples, network, learning_rate, num_epochs):
    """
    Returns a network after training it

    Args:
        examples: Array of examples in the form of input and output values
            e.g [[num_examples x num_inputs],[num_examples x num_outputs]]
        network: 2 dimensional array of nodes
            e.g. [[num_nodes_per_layer] x num_layers]
        learning_rate: Floating point number
        num_epochs: Integer

    Returns:
        network: 2 dimensional array of nodes
            e.g. [[num_nodes_per_layer] x num_layers]
    """
    inputs, outputs = examples
    input_nodes = network[0]
    output_nodes = network[2]

    for _ in range(num_epochs):
        # Loop over every example
        for input, output in zip(inputs, outputs):
            load_input_values(input_nodes, input)
            # Forward propagate values from input layer to output layer
            forward_propagate(network)
            # Initialize delta
            delta = [[] for layer in network]
            # Compute delta for output layer
            for i, node in enumerate(output_nodes):
                delta[2].append(derivative_of_sigmoid(node.inputValue) * (output[i] - node.value))
            # Propagate error backwards
            back_propagate(network, delta)
            # Update weights using delta
            update_weights(network, delta, learning_rate)
        print("Completed epoch " + str(_ + 1))
    return network

"""Functions used in back_prop_learning algorithm"""

def load_input_values(input_nodes, input_values):
    """ Copy input vector of one training example to input nodes of network """
    for i, node in enumerate(input_nodes):
        node.value = input_values[i]

def forward_propagate(network):
    """ Forward propagate values from input layer to output layer """
    for layer in network[1:]:
        for node in layer:
            node.inputValue = -1 * node.weights[0]
            for i, input_node in enumerate(node.inputs):
                node.inputValue += input_node.value * node.weights[i+1]
            node.value = sigmoid(node.inputValue)

def back_propagate(network, delta):
    """Back propagate error from output layer to hidden layers"""
    num_layers = len(network)
    for i in range(num_layers - 2, 0, -1):
        layer = network[i]
        next_layer = network[i+1]
        weights = []
        num_nodes = len(layer)
        # Compute delta for the hidden layer
        for j in range(num_nodes):
            # Get weights to compute delta
            weights.append([node.weights[j+1] for node in next_layer])
            # Sum of the multiplied deltas and weights
            tmp_sum = 0
            for k, weight in enumerate(weights[j]):
                tmp_sum += weight * delta[i+1][k]
            delta[i].append(derivative_of_sigmoid(layer[j].inputValue) * tmp_sum)

def update_weights(network, delta, learning_rate):
    """Update weights of the entire network using delta"""
    num_layers = len(network)
    for i in range(1, num_layers):
        layer = network[i]
        prev_layer_node_values = [-1] + [node.value for node in network[i-1]]
        for j, node in enumerate(layer):
            update_values = [learning_rate * delta[i][j] * node_value
                             for node_value in prev_layer_node_values]
            node.weights = [node.weights[k] + update_values[k]
                            for k, weight in enumerate(node.weights)]
