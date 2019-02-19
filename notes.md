build an ANN extend the perceptron into an ANN

recognize characters, A-Z

additional notes on OCR can be found in week 1 lecture presentation

the data set contains in the first column the letter, followed by some attributes already figured out, e.g. position, width, height, etc.

try normalizing with respect to all, but try to understand the meaning of each value, makes more sense to normalize each column by its max

17 inputs, 2 hidden layers, then 26 outputs, 26 neurons on the output, sigmoid activation because we want real values out values btwn. 0-1 and then take the soft max (index of the highest value), this index is the letter value (0 = A, 1 = B, etc.)

build a dictionary

1. convert the letters in the first col to numbers 0-25

build a 26x26 array, if you feed it A and it guesses A add 1 to (0,0) (A, A)
if you feed it A and it says C then add 1 to (0, 2) (A, C)
