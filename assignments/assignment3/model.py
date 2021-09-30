import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.conv_1 = ConvolutionalLayer(input_shape[2], conv1_channels, 3, 0)
        self.conv_2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 0)
        self.max_pool_1 = MaxPoolingLayer(4, 2)
        self.max_pool_2 = MaxPoolingLayer(4, 2)
        self.flat = Flattener()
        self.relu_1 = ReLULayer()
        self.relu_2 = ReLULayer()
        self.fc = FullyConnectedLayer(50, n_output_classes)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        self.fc.W.grad = np.zeros_like(self.fc.W.grad)
        self.fc.B.grad = np.zeros_like(self.fc.B.grad)
        self.conv_2.W.grad = np.zeros_like(self.conv_2.W.grad)
        self.conv_2.B.grad = np.zeros_like(self.conv_2.B.grad)
        self.conv_1.W.grad = np.zeros_like(self.conv_1.W.grad)
        self.conv_1.B.grad = np.zeros_like(self.conv_1.B.grad)
        # d_out = 0
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        output = self.conv_1.forward(X)
        output = self.relu_1.forward(output)
        output = self.max_pool_1.forward(output)
        output = self.conv_2.forward(output)
        output = self.relu_2.forward(output)
        output = self.max_pool_2.forward(output)
        output = self.flat.forward(output)
        # print('shape: ', output.shape)
        output = self.fc.forward(output)
        loss, d_out = softmax_with_cross_entropy(output, y)

        d_out = self.fc.backward(d_out)
        d_out = self.flat.backward(d_out)
        d_out = self.max_pool_2.backward(d_out)
        d_out = self.relu_2.backward(d_out)
        d_out = self.conv_2.backward(d_out)
        d_out = self.max_pool_1.backward(d_out)
        d_out = self.relu_1.backward(d_out)
        d_out = self.conv_1.backward(d_out)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        output = self.conv_1.forward(X)
        output = self.relu_1.forward(output)
        output = self.max_pool_1.forward(output)
        output = self.conv_2.forward(output)
        output = self.relu_2.forward(output)
        output = self.max_pool_2.forward(output)
        output = self.flat.forward(output)
        # print('shape: ', output.shape)
        output = self.fc.forward(output)

        pred = np.argmax(output, 1)

        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result["CONV_2 W"] = self.conv_2.W
        result["CONV_2 B"] = self.conv_2.B
        result["CONV_1 W"] = self.conv_1.W
        result["CONV_1 B"] = self.conv_1.B
        result["FC W"] = self.fc.W
        result["FC B"] = self.fc.B

        return result
