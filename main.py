"""Main program"""
from neural_network_programs import (
    training_program, testing_program
)

def main():
    """ Executes training or testing program """
    while True:
        print("Program Options:")
        print("0  |  Training Program\n1  |  Testing Program")
        mode = input("To choose a program, type 0 or 1 and hit enter:\n")
        if mode == '0':
            print("Entered Training Program\n------------------------")
            training_program()
        if mode == '1':
            print("Entered Testing Program\n------------------------")
            testing_program()

main()
