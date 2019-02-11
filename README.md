Parameters like MAXDEPTH, NUMTREES, and ITERS (number of iterations for cross validation) are all at the top of hw5.py.

To just run random forest run the code as is.

To run just one decision tree with every feature considered comment out lines 65-67 and comment in line 69 and assign NUMTREES = 1.

To run the code on the test set call test() at the bottom. Currently the code has validation() at the bottom which sets aside 1/5 of the training set as a validation set and prints the error on the validation set.
