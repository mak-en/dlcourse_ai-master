import numpy as np
from numpy.core.numeric import zeros_like


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

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
    '''
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
    '''
    # TODO copy from the previous assignment
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        relu_output = None
        relu = lambda x: x * (x > 0).astype(float)
        # relu_output = np.where(X > 0, X, X * 0.01)  
        relu_output = relu(X)
        self.cache = X
        
        return relu_output

    def backward(self, d_out):
        # TODO copy from the previous assignment
        dx, x = None, self.cache
        dx = d_out * (self.cache > 0)

        return dx

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding


    def forward(self, X):

        batch_size, height, width, channels = X.shape
        # print(X.shape)
        
        if self.padding:
            X_pad = np.zeros((X.shape[0], X.shape[1]+2*self.padding, X.shape[2]+2*self.padding, X.shape[3]))
            for img in range(X.shape[0]):
                for ch in range(X.shape[-1]):
                    # print(X[img, :, :, ch].shape)
                    X_pad[img,:,:,ch] = np.pad(X[img,:,:,ch], ((self.padding,self.padding),(self.padding,self.padding)), 'constant')

            X = X_pad

        self.X = X

        out_height = (height - self.filter_size + 2 * self.padding) + 1
        out_width = (width - self.filter_size + 2 * self.padding) + 1

        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # print(output.shape)
        
        W = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.W.value.shape[-1])
        # print('------------------')
        # print(f'X: {X}','\n', f'W: {W}\n')
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                output[:, y, x, :] = np.dot(X[:, y:(y+self.filter_size), x:(x+self.filter_size), :].reshape(X.shape[0], \
                self.filter_size*self.filter_size*self.in_channels), W) + self.B.value

        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # print(d_out.shape)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_input = np.zeros(self.X.shape)
        # print(d_input.shape)
        # self.W.grad = np.zeros_like(self.W.grad)
        # self.B.grad = np.zeros_like(self.B.grad)
        W = self.W.value.reshape(self.filter_size*self.filter_size*self.in_channels, self.W.value.shape[-1])

        # # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)

                d_input[:, y:(y+self.filter_size), x:(x+self.filter_size), :] += \
                np.dot(d_out[:, y, x, :], W.T).reshape(batch_size, self.filter_size, self.filter_size, channels)

                self.W.grad += np.dot(self.X[:, y:(y+self.filter_size), x:(x+self.filter_size), :].reshape(self.X.shape[0], \
                    self.filter_size*self.filter_size*self.in_channels).T, d_out[:, y, x, :]).reshape(self.filter_size, self.filter_size, \
                    self.in_channels, self.out_channels)

        self.B.grad = np.sum(d_out, axis=(0,1,2)).reshape(out_channels)

        if self.padding:
            d_input = np.delete(d_input, np.s_[0:self.padding], 1)
            d_input = np.delete(d_input, np.s_[-self.padding:], 1)
            d_input = np.delete(d_input, np.s_[0:self.padding], 2)
            d_input = np.delete(d_input, np.s_[-self.padding:], 2)

        # print(d_input.shape)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.mask = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        # print(height, width)
        out_height = ((height - self.pool_size) / self.stride + 1)
        out_width = ((width - self.pool_size) / self.stride + 1)
        number_height = str(out_height-int(out_height)).split(".")[1] 
        number_width = str(out_width-int(out_width)).split(".")[1]

        if number_height != "0" or number_width != "0":
          raise TypeError("The output of MaxPooling can only be integer numbers")
        else:
          out_height = int(out_height)
          out_width = int(out_width)

        output = np.zeros((batch_size, out_height, out_width, channels))
        self.mask = np.zeros_like(X)

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location

                tmp = X[:, (y*self.stride):((y*self.stride)+self.pool_size), \
                                    (x*self.stride):((x*self.stride)+self.pool_size), :]
                
                output[:, y, x, :] = tmp.max(axis=(1,2))

                # print(tmp.shape)
                # print(output[:, y, x, :])
                # print(output[:, y, x, :].shape)
                # print(tmp.argmax(axis=(2)))
                maxs = output[:, y, x, :].repeat(self.pool_size, axis=0).repeat(self.pool_size, axis=0)
                # print(maxs)
                maxs = maxs.reshape(tmp.shape, order='C')

                # print(maxs)

                # print(np.equal(tmp, maxs).astype(int))

                # print(np.where(tmp == tmp.max(axis=(1,2)), 1, 0))


                self.mask[:, (y*self.stride):((y*self.stride)+self.pool_size), \
                                    (x*self.stride):((x*self.stride)+self.pool_size), :] = np.equal(tmp, maxs).astype(int)

        # print(self.X.shape,'\n')
        # print('--------------\n')
        # print("mask: ", self.mask.shape)
        # print("output: ", output.shape)
       
        return output

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape

        # print("d_out: ", d_out.shape)
        
        # print(self.mask.shape)
        dX = zeros_like(self.X)
        # print("dX: ", dX.shape)

        for y in range(d_out.shape[1]):
          for x in range(d_out.shape[2]):
            # print("d_out x y: ", d_out[:, y, x, :].shape)
            
            tmp = d_out[:, y, x, :].repeat(self.pool_size, axis=0).repeat(self.pool_size, axis=0)
            tmp = tmp.reshape(dX[:, (y*self.stride):((y*self.stride)+self.pool_size), \
              (x*self.stride):((x*self.stride)+self.pool_size), :].shape, order='C')
            # print("d_out x y 2: ", tmp.shape)

            dX[:, (y*self.stride):((y*self.stride)+self.pool_size), \
              (x*self.stride):((x*self.stride)+self.pool_size), :] += tmp


        # dX = d_out.repeat(self.pool_size, axis=1).repeat(self.pool_size, axis=2)
        dX = np.multiply(dX, self.mask)

        return dX

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape(batch_size, height*width*channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}