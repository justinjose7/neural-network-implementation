""" Training program, test program, and functions used by both """
import random

from back_prop_algorithm import (
    back_prop_learning, load_input_values, forward_propagate
)

class Node:
    """
    Simple node class used to represent node in neural network

    Attributes:
        weights: Array of weights of edges connecting nodes of previous
                layer to this node
        inputs: Array of nodes of previous layer which connect to this node
        value: Floating point value of node (value after activation function
                is applied to sum of the input values to the node)
        input_value: Floating point value of the sum of input values
                to the node
    """
    def __init__(self):
        self.weights = []
        self.inputs = []
        self.value = 0
        self.input_value = 0

def generate_network(filename):
    """
    Returns a network generated from formatted initial weights file (.init)

    Args:
        filename: String referring to name of weights file located in folder
    Returns:
        network: 2 dimensional array of nodes
            e.g. [[num_nodes_per_layer] x num_layers]

    """
    try:
        file = open(filename, "r")
    except IOError:
        print("No such file: '" + filename + "'")
        quit()
    # Get layer sizes from first line of file
    layer_sizes = []
    for size in next(file).split():
        layer_sizes.append(int(size))
    # Initialize network with correct number of nodes per layer
    network = []
    for size in layer_sizes:
        layer = []
        for i in range(size):
            layer.append(Node())
        network.append(layer)
    # For each node specify node inputs from previous layer and the
    # weights associated with each input given in the .init file
    for i in range(1, len(network)):
        for node in network[i]:
            weights = []
            for weight in next(file).split():
                weights.append(float(weight))
            node.weights = weights
            for input_node in network[i-1]:
                node.inputs.append(input_node)
    file.close()
    return network

def generate_new_network(num_inputs, num_hidden_nodes, num_outputs):
    """
    Returns a random network of nodes given layer sizes

    Args:
        num_inputs: Integer number of input nodes
        num_hidden_nodes: Integer number of hidden layer nodes
        num_outputs: Integer number of output nodes

    Returns:
        network: 2 dimensional array of nodes
            e.g. [[num_nodes_per_layer] x num_layers]

    """
    # Get layer sizes from first line of file
    layer_sizes = [num_inputs, num_hidden_nodes, num_outputs]
    # Initialize network with correct number of nodes per layer
    network = []
    for size in layer_sizes:
        layer = []
        for i in range(size):
            layer.append(Node())
        network.append(layer)
    # For each node specify node inputs from previous layer and the
    # weights associated with each input given in the .init file
    for i in range(1, len(network)):
        for node in network[i]:
            node.weights = [random.uniform(0, 1) for _ in range(layer_sizes[i-1] + 1)]
            for input_node in network[i-1]:
                node.inputs.append(input_node)
    return network

def get_examples(filename):
    """
    Returns inputs and outputs for examples from formatted training file (.train)

    Args:
        filename: String referring to name of training file located in folder
    Returns:
        inputs, outputs: Array of examples in the form of input and output values
            e.g [[num_examples x num_inputs],[num_examples x num_outputs]]

    """
    try:
        file = open(filename, "r")
    except IOError:
        print("No such file: '" + filename + "'")
        quit()

    num_examples, num_inputs, num_outputs = [int(x) for x in next(file).split()]
    examples = []
    for _ in range(num_examples):
        examples.append([float(x) for x in next(file).split()])
    file.close()
    inputs = []
    outputs = []
    for example in examples:
        inputs.append(example[:num_inputs])
        outputs.append([int(e) for e in example[num_inputs:]])

    return inputs, outputs

def write_network_to_file(filename, network):
    """
    Writes the network (layer sizes + weights) to a file

    Args:
        filename: String referring to name of weights file located in folder
        network: 2 dimensional array of nodes
            e.g. [[num_nodes_per_layer] x num_layers]
    """
    file = open(filename, "w+")
    num_inputs = len(network[0])
    num_hidden_nodes = len(network[1])
    num_outputs = len(network[2])
    file.write(str(num_inputs) + " " + str(num_hidden_nodes) + " " + str(num_outputs) + "\n")
    for i in range(1, 3):
        for node in network[i]:
            weights = ""
            for weight in node.weights:
                weights += "{0:.3f}".format(weight) + " "
            file.write(weights.strip() +  "\n")
    file.close()

def training_program():
    """Lets user specify initial weights file and a training file, generates
    the network and writes the network to user specified file"""
    weights_file = input("Enter weights filename:\n")
    network = generate_network(weights_file)
    training_file = input("Enter training filename:\n")
    examples = get_examples(training_file)
    output_file = input("Enter output filename:\n")
    num_epochs = int(input("Enter the number of epochs:\n"))
    learning_rate = float(input("Enter the learning rate:\n"))
    trained_network = back_prop_learning(examples, network, learning_rate, num_epochs)
    write_network_to_file(output_file, trained_network)

def threshold_function(value):
    """Returns 0 or 1 depending on whether value is >= 0.5"""
    if value < 0.5:
        return 0
    else:
        return 1

def predict(network, examples):
    """Returns confusion matrix values A, B, C, D given test examples"""
    inputs, outputs = examples
    input_nodes = network[0]
    output_nodes = network[2]
    num_outputs = len(output_nodes)
    a = [0 for i in range(num_outputs)]
    b = [0 for i in range(num_outputs)]
    c = [0 for i in range(num_outputs)]
    d = [0 for i in range(num_outputs)]
    for input, output in zip(inputs, outputs):
        load_input_values(input_nodes, input)
        forward_propagate(network)
        predicted_output = [threshold_function(output_node.value)
                            for output_node in output_nodes]

        for i, prediction in enumerate(predicted_output):
            if prediction == 1:
                if output[i] == 1:
                    a[i] += 1
                if output[i] == 0:
                    b[i] += 1
            if prediction == 0:
                if output[i] == 1:
                    c[i] += 1
                if output[i] == 0:
                    d[i] += 1
    return a, b, c, d

def write_statistics_to_file(filename, a, b, c, d):
    """
    Computes additional statistics given A, B, C, D confusion matrix values
    and writes metric values to user specified file.
    """
    num_outputs = len(a)
    overall_accuracy = [(a[i] + d[i]) / (a[i] + b[i]+ c[i]+ d[i])
                        for i in range(num_outputs)]
    precision = [a[i] / (a[i] + b[i]) for i in range(num_outputs)]
    recall = [a[i] / (a[i] + c[i]) for i in range(num_outputs)]
    f_1 = [(2 * precision[i] * recall[i]) / (precision[i] + recall[i])
           for i in range(num_outputs)]

    a_global = sum(a)
    b_global = sum(b)
    c_global = sum(c)
    d_global = sum(d)

    micro_avg_overall_accuracy = (a_global + d_global) / (a_global + b_global
                                                          + c_global + d_global)
    micro_avg_precision = a_global / (a_global + b_global)
    micro_avg_recall = a_global / (a_global + c_global)
    micro_avg_f_1 = (2 * micro_avg_precision * micro_avg_recall) / (micro_avg_precision
                                                                    + micro_avg_recall)

    macro_avg_overall_accuracy = sum(overall_accuracy)/len(overall_accuracy)
    macro_avg_precision = sum(precision)/len(precision)
    macro_avg_recall = sum(recall)/len(recall)
    macro_avg_f_1 = (2 * macro_avg_precision * macro_avg_recall) / (macro_avg_precision
                                                                    + macro_avg_recall)
    file = open(filename, "w+")
    for i in range(num_outputs):
        file.write(str(a[i]) + " " + str(b[i]) + " " + str(c[i]) + " " + str(d[i]) + " "
                   + "{0:.3f}".format(overall_accuracy[i]) + " "
                   + "{0:.3f}".format(precision[i]) + " "
                   + "{0:.3f}".format(recall[i]) + " "
                   + "{0:.3f}".format(f_1[i]) + "\n");
    file.write("{0:.3f}".format(micro_avg_overall_accuracy) + " "
               + "{0:.3f}".format(micro_avg_precision) + " "
               + "{0:.3f}".format(micro_avg_recall) + " "
               + "{0:.3f}".format(micro_avg_f_1) + "\n")
    file.write("{0:.3f}".format(macro_avg_overall_accuracy) + " "
               + "{0:.3f}".format(macro_avg_precision) + " "
               + "{0:.3f}".format(macro_avg_recall) + " "
               + "{0:.3f}".format(macro_avg_f_1) + "\n");
    file.close()

def testing_program():
    """Lets user specify initial weights file and a testing file, generates
    predictions and writes associated statistics to user specified file"""
    weights_file = input("Enter weights filename:\n")
    network = generate_network(weights_file)
    testing_file = input("Enter testing filename:\n")
    examples = get_examples(testing_file)
    output_file = input("Enter output filename:\n")
    a, b, c, d = predict(network, examples)
    write_statistics_to_file(output_file, a, b, c, d)
