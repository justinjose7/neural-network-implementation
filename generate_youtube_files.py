"""Program to generate youtube.init, youtube.trained, and youtube.results"""
from back_prop_algorithm import (
    back_prop_learning
)
from neural_network_programs import (
    generate_new_network, write_network_to_file, generate_network, get_examples,
    predict, write_statistics_to_file
)

def main():
    """
    Generates youtube.init, youtube.trained, and youtube.results
    Note: To use, you must move youtube.init, youtube.train and youtube.test into
    this directory

    """
    # Commented out below lines so the youtube.init file (uniform randomly generated weights)
    # I used for my training and testing isn't overridden
    #
    # new_network = generate_new_network(4, 2, 1)
    # init_file = "youtube.init"
    # write_network_to_file(init_file, new_network)
    weights_file = "youtube.init"
    training_file = "youtube.train"
    trained_file = "youtube.1.500.trained"
    testing_file = "youtube.test"
    result_file = "youtube.1.500.results"
    num_epochs = 500
    learning_rate = 0.1
    network = generate_network(weights_file)
    examples = get_examples(training_file)
    trained_network = back_prop_learning(examples, network, learning_rate, num_epochs)
    write_network_to_file(trained_file, trained_network)
    examples = get_examples(testing_file)
    a, b, c, d = predict(trained_network, examples)
    write_statistics_to_file(result_file, a, b, c, d)

main()
