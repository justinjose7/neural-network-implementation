# Neural-Network-Implementation
Training and testing programs for a neural network program written for ECE 469 Artificial Intelligence

Implementation involves an input layer, a single hidden layer, and an output layer.

## How to run the program

```
python3 main.py
```
Then enter 0 or 1 after program begins to select either the training or testing program.

When your neural network training program is executed, it should prompt the user for the names of three text files representing the initial neural network, a training set, and an output file; one positive integer representing the number of epochs; and one floating-point value representing the learning rate. The first text file should contain the representation of the neural network before training (i.e., it will specify the size of each layer and the initial weights of the network). 

## File Formatting Specifications

### Initial weights file

    • The first line will contain three integers, separated by single spaces, representing the number of input nodes (Ni), the number of hidden nodes (Nh), and the number of output nodes (No).
    • The next Nh lines specify the weights of edges pointing from input nodes to hidden nodes. The first of these lines specifies the weights of edges entering the first hidden node; the second line specifies the weights of edges entering the second hidden node; etc. Each of these Nh lines specifies Ni + 1 weights, which will be floating-point numbers separated by single spaces. These weights include the bias weight which is attached to a fixed input that always has its activation set to -1. (Note that the fixed input should be -1, not +1, so we are not following the convention of the current edition of the textbook.) Using a fixed input of -1 makes the bias weight equivalent to a threshold to which the total of the true weighted input can be compared. For each hidden node, the first weight represents the bias weight, and the next Ni weights represent the weights of edges from the input nodes to the hidden node.
    • The next No lines specify the weights of edges pointing from hidden nodes to output nodes. The first of these lines specifies the weights of edges entering the first output node; the second line specifies the weights of edges entering the second output node; etc. Each of these No lines specifies Nh + 1 weights, which will be floating-point numbers separated by single spaces. These weights include the bias weight which is attached to a fixed input that always has its activation set to -1. For each output node, the first weight represents the bias weight, and the next Nh weights represent the weights of edges from the hidden nodes to the output node.



### Training set file
    • The first line of this file contains three integers separated by spaces: the number of training examples, Ni, and No (I guarantee that Ni and No will match the neural network being trained). 
    • Every other line of the file specifies a single example and contains Ni floating-point inputs (the values of the example's input attributes) followed by No Boolean outputs (each is 0 or 1). 

### Other notes regarding training program
    • The number of epochs specifies how many times the outer loop of the pseudo code (the repeat...until loop in the pseudo code) should iterate. This is the second simplification to the given pseudo code; instead of training until some stopping criterion is satisfied, training will proceed for a specified number of epochs. (Remember that each epoch loops through the entire training set, updating the weights of the network for every training example.) The learning rate is a floating-point value that determines how fast weights are updated; values that are two big will cause weights to overshoot their appropriate values, and values that are too low will cause the learning to be inefficient, requiring too many epochs before reaching a near-optimal state. The third text file, to be created by the training program, should have exactly the same format as the first text file, but the weights contained in this file should be the learned weights of the trained neural network.

When your neural network testing program is executed, it should prompt the user for the names of three text files. The first text file contains the representation of a neural network (presumably one that has already been trained) using the previously described format. The second text file specifies the test set for the neural network; i.e., this file contains testing examples with which to test the network. The format of this file is the same as the format for the file that specifies a training set for a neural network (as has been described above). The program should iterate through every test example, and for each, it should compute the outputs according to the specified neural network and compare them to the expected outputs. When comparing actual outputs to expected outputs during testing, all actual outputs (i.e., the activations of the output nodes) that are ≥ 0.5 should be rounded up to 1, and all outputs that are < 0.5 should be rounded down to 0. (Note that this rounding should not occur during training when computing errors of output nodes. The rounding only occurs during testing when determining whether or not an output of the neural network is correct.) As testing proceeds, your program will need to keep track of various counts in order to compute several metrics that will soon be explained. The third text file should store the results of these metrics with a format that will soon be specified.

## Dataset
The custom dataset attached is a YouTube Trending / Non-Trending dataset. Input features are view count, like count, dislike count, and comment count. There is one output feature which is simply 1 or 0 (representing trending or nontrending). 

The YouTube API was used to generate the data required for the dataset. The scripts written to generate the dataset can be found in the 'youtube-dataset-misc-files' folder.

## Credits
The YouTube Trending / Non-Trending dataset was created with the help of Jonathan Mathai and Richu Jacob.
