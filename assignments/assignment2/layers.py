import numpy as np
from numpy.core.fromnumeric import shape


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    if predictions.ndim == 1 :
      epowa = np.exp(predictions-np.max(predictions))
      return epowa/np.sum(epowa)
    else:
      epowa = np.exp(predictions-np.max(predictions,axis=1,keepdims=True))
      return epowa/np.sum(epowa,axis=1,keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim == 1:
      trues = np.zeros(probs.shape)
      trues[target_index] = 1
      return -np.sum(trues * np.log(probs))
    elif target_index.ndim == 1:
      trues = np.eye(probs.shape[1])[target_index]
      return -np.sum(trues * np.log(probs)) / probs.shape[0]
    else:
      trues = np.eye(probs.shape[1])[target_index[:, 0]]
      return -np.sum(trues * np.log(probs)) / probs.shape[0]


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probes = softmax(preds)
    loss = cross_entropy_loss(probes, target_index)
    if preds.ndim == 1:
      trues = np.zeros(preds.shape)
      trues[target_index] = 1
      dprediction = probes - trues
    elif target_index.ndim == 1:
      trues = np.eye(probes.shape[1])[target_index]
      dprediction = (probes - trues) / probes.shape[0]
    else:
      trues = np.eye(probes.shape[1])[target_index[:, 0]]
      dprediction = (probes - trues) / probes.shape[0]

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        relu_output = None
        relu = lambda x: x * (x > 0).astype(float)
        # relu_output = np.where(X > 0, X, X * 0.01)  
        relu_output = relu(X)
        self.cache = X
        
        return relu_output
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        dx, x = None, self.cache
        dx = d_out * (self.cache > 0)

        return dx
        

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        # self.W = Param(np.random.randn(n_input) * np.sqrt(2.0/n_input))
        # self.B = Param(0 * np.random.randn(1, n_output))

        self.B = Param(np.zeros((1, n_output)))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        # print(self.B.value, '\n')
        # print((np.dot(X, self.W.value) + self.B.value).shape, '\n')
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        # print(np.sum(d_out, axis=0, keepdims=True), '\n')

        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        # print(f'grad B {self.B.grad}\n')
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
