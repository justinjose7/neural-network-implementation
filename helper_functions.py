""" Helper functions for project """
import math

def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + math.exp(-x))

def derivative_of_sigmoid(x):
    """Derivative of sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))
