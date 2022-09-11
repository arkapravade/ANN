# ANN using C++

The Multilayered Feed-forward Neural Network (MLFFNN) was created from scratch using C++,
allowing user to test any number of patterns having however many features or outputs
the dataset  possesses. The user can also set the number of hidden layer, maximum number 
of iterations they want to run the code for and the number of patterns they want to
reserve for testing.

# ANN using Keras

A more advanced and capable version of the Artificial Neuron Network (ANN), was developed
using Keras in python which was tested using Thermodynamic Steam Table containing over 250
patters. In this scope, pressure and temperature were considered as features/iunput pareameters,
and internal energy & enthalpy for liquid & gas phases were taken as four output values.
After scaling the data, the model was developed using three hidden layers each having
500 nodes, with Rectified Linear Unit (ReLU) as activation function, 'Adam' as optimizer and
Mean Squared Error (MSE) as loss function.

As the dataset was not very large, 20% of the data were reserved for testing. After training
the model, the Root Mean Squared Error (RMSE) turned out to be around 16.5. A manual sample
was also tested to validate the model.
