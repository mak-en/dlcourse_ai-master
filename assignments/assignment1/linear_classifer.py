from matplotlib.pyplot import axis
import numpy as np
from numpy.core.fromnumeric import repeat
from gradient_check import check_gradient


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


def softmax_with_cross_entropy(predictions, target_index):
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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probes = softmax(predictions)
    loss = cross_entropy_loss(probes, target_index)
    if predictions.ndim == 1:
      trues = np.zeros(predictions.shape)
      trues[target_index] = 1
      dprediction = probes - trues
    elif target_index.ndim == 1:
      trues = np.eye(probes.shape[1])[target_index]
      dprediction = (probes - trues) / probes.shape[0]
    else:
      trues = np.eye(probes.shape[1])[target_index[:, 0]]
      dprediction = (probes - trues) / probes.shape[0]
    
    
    return loss, dprediction



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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    return loss, grad

    
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss = softmax_with_cross_entropy(predictions, target_index)[0]
    dW = np.dot(X.T, softmax_with_cross_entropy(predictions, target_index)[1])

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    return loss, dW
    raise Exception("Not implemented!")
    
    


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            loss = 0
            for batch in batches_indices:
              # print(f"batch: {batch}, batch X: {X[batch]}, batch y {y[batch]}")
              batch_loss = linear_softmax(X[batch], self.W, y[batch])[0] + l2_regularization(self.W, reg)[0]
              grad = linear_softmax(X[batch], self.W, y[batch])[1] + l2_regularization(self.W, reg)[1]
              self.W += -learning_rate * grad
              loss += batch_loss

            loss = loss / len(batches_indices)

            loss_history.append(loss)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!


            

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

            # raise Exception("Not implemented!")

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        y_pred = np.argmax(np.dot(X, self.W), 1)

        return y_pred
                                                          

                
