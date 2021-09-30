import numpy as np
from numpy.core.fromnumeric import shape

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.l1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.l2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.l1.W.grad = np.zeros_like(self.l1.W.value)
        self.l1.B.grad = np.zeros_like(self.l1.B.value)
        self.l2.W.grad = np.zeros_like(self.l2.W.value)
        self.l2.B.grad = np.zeros_like(self.l2.B.value)
        
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        output = self.l1.forward(X)
        output = self.relu.forward(output)
        output = self.l2.forward(output)
        

        loss, d_out = softmax_with_cross_entropy(output, y)
        loss += l2_regularization(self.l1.W.value, reg_strength=self.reg)[0] + \
                l2_regularization(self.l2.W.value, reg_strength=self.reg)[0]

        d_out = self.l2.backward(d_out)
        d_out = self.relu.backward(d_out)
        self.l1.backward(d_out)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        self.l1.W.grad += l2_regularization(self.l1.W.value, self.reg)[1]
        self.l2.W.grad += l2_regularization(self.l2.W.value, self.reg)[1]

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        # print(self.l1.W.value, '\n')
        # print(f'X: {X}\n')
        output = self.l1.forward(X)
        # print(f'l1: {output}\n')
        output = self.relu.forward(output)
        # print(f'relu: {output}\n')
        output = self.l2.forward(output)
        # print(f'l2 W: {self.l2.W.value}\n')
        # print(f'l2 B: {self.l2.B.value}\n')
        # print(f'l1 B: {self.l1.B.value}\n')
        # print(f'l2: {output}\n')
        

        # print(f'relu: {output[0]}')
        pred = np.argmax(output, 1)
        # print(pred, '\n')
        
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        result['W_l1'] = self.l1.W
        result['B_l1'] = self.l1.B
        result['W_l2'] = self.l2.W
        result['B_l2'] = self.l2.B

        return result
