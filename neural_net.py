import numpy as np
import matplotlib.pyplot as plt

class MLPNet(object):

  """
  In this class we implement a MLP neural network. 
  H: hidden layer size
  N: input size
  D: Number of features
  C: class
  Loss Function: Softmax
  Regularization: L2 norm
  Activation Function: ReLU
  
  """
  def __init__(self, D, H, C, std=1e-4):#N, H, output_size, std=1e-4):
    """
    In this part we initialize the model as below:
    weights are initialize with small random value and biases are initialized with zero value. 
    these values are stored in the self.p_net as dictionary
    """
    np.random.seed(1)
    self.p_net = {}
    self.p_net['W1'] = std * np.random.randn(D, H) * np.sqrt(2./D)#N, H)
    self.p_net['b1'] = np.zeros(H)
    self.p_net['W2'] = std * np.random.randn(H, C) * np.sqrt(2./H)#N, H)
    self.p_net['b2'] = np.zeros(C)#output_size)

  def loss(self, X, y=None, reg=0.0):

    """
      calculate the loss and its gradients for network:
      our inputs are:
        X: N*D matrix 
        y: training labels

      Returns:
      if y is empty :
        -return score matrix with shape (N,C) .each element of this matrix shows score for class c on input X[i]
      otherwise:
        -return a tuple of loss and gradient.
    """
    Weight2, bias2 = self.p_net['W2'], self.p_net['b2']
    Weight1, bias1 = self.p_net['W1'], self.p_net['b1']
    N, D = X.shape

    # forward pass
    scores = None
    #############################################################################
    # calculate output of each neurons
    # store results in the scores variable.                                                          
    #############################################################################
    Z1 = np.dot(X, Weight1) + bias1
    A1 = np.maximum(0,Z1) # ReLU acticvation
    Z2 = np.dot(A1, Weight2) + bias2
    
    A2_softmax = np.exp(Z2)
    A2_softmax = A2_softmax / np.sum(A2_softmax, axis = 1).reshape(-1,1)
    
    scores = A2_softmax
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    if y is None:
      return scores

    # fill loss function.
    loss = None
    ############################################################################# 
    # loss = data loss + L2 regularization                                      
    #############################################################################
    one_hot_dim = Weight2.shape[1]
    Y = np.eye(one_hot_dim)[y]
    cross_entropy_loss = -(1./N) * np.sum(np.multiply(Y, np.log(A2_softmax)))
    
    L2_regularization_cost = (1./N) * (reg/2) * ( np.sum(np.square(Weight1)) + np.sum(np.square(Weight2)) )
    loss = cross_entropy_loss +  L2_regularization_cost
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # calculate gradients
    gradient = {}
    #############################################################################
    # store derivation of network's parameters(W and b) in the gradient
    # as dictionary structure
    #############################################################################
    dZ2 = (A2_softmax - Y) #init
    
    dW2 = 1./N * np.dot(A1.T, dZ2) + (reg/N)*Weight2
    db2 = 1./N * np.sum(dZ2, axis=0, keepdims = True)
    
    dA1 = np.dot(dZ2, Weight2.T)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
       
    dW1 = 1./N * np.dot(X.T, dZ1) + (reg/N)*Weight1
    db1 = 1./N * np.sum(dZ1, axis=0, keepdims = True)
    
    gradient = {"W2": dW2, "b2": db2, 
                "W1": dW1, "b1": db1}

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, gradient


  def train(self, X, y, X_val, y_val,
            alpha=1e-3, alpha_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=100, verbose=False):

    """
    We want to train this network with stochastic gradient descent.
    Our inputs are:

    - X: array of shape (N,D) for training data.
    - y: training labels.
    - X_val: validation data.
    - y_val: validation labels.
    - alpha: learning rate
    - alpha_decay: This factor used to decay the learning rate after each epoch
    - reg: That shows regularization .
    - num_iters: Number of epoch 
    - batch_size: Size of each batch

    """
    num_train = X.shape[0]
    import math
    iteration = int(max(num_train / batch_size, 1))

    loss_train = []
    train_acc = []
    va_acc = [] 
    
    one_hot_dim = self.p_net['W2'].shape[1]
    seed = 20
    
    for it in range(num_iters): # +1): for printing result purpose (%10 last result)
        seed = seed + 1
        permutation = list(np.random.permutation(num_train))
        shuffled_X = X[permutation, :]
        shuffled_y = y[permutation]

        for i in range(iteration):
            data_batch = None
            label_batch = None
            #########################################################################
            # create a random batch of data and labels for training store 
            # them into data_batch and label_batch  
            #########################################################################
            data_batch = shuffled_X[i*batch_size:(i+1)*batch_size, :]
            label_batch = shuffled_y[i*batch_size:(i+1)*batch_size]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # calculate loss and gradients
            loss, gradient = self.loss(data_batch, y=label_batch, reg=reg)
            loss_train.append(loss)

            #########################################################################
            # update weights and biases which stored in the slef.p_net regarding 
            # to gradient dictionary.
            #########################################################################
            self.p_net['W1'] = self.p_net['W1'] - alpha * gradient['W1']
            self.p_net['b1'] = self.p_net['b1'] - alpha * gradient['b1']
            self.p_net['W2'] = self.p_net['W2'] - alpha * gradient['W2']
            self.p_net['b2'] = self.p_net['b2'] - alpha * gradient['b2']

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

        if (it % 1 == 0) and (verbose == True): # % 100
            #print ('iteration %d / %d: loss %f' % (it, num_iters, loss))
            print ('Epoch %d / %d: loss %f, Accuracy %f ' % (it, num_iters, loss
                                                                      , (self.predict(data_batch) == label_batch).mean() ))

        if it % 1 == 0: #% iteration
            # Check accuracy
            #train_acc = (self.predict(data_batch) == label_batch).mean() 
            train_accuracy = (self.predict(data_batch) == label_batch).mean() 
            #val_acc = (self.predict(X_val) == y_val).mean()
            val_accuracy = (self.predict(X_val) == y_val).mean()
            train_acc.append(train_accuracy)#train_acc)
            va_acc.append(val_accuracy)#va_(...)val_acc)
            alpha *= alpha_decay
            #print('train_acc' + str(train_accuracy))
        

    return {
        'loss_train': loss_train,
        'train_acc': train_acc,
        'va_acc': va_acc,
    }

  def predict(self, X):

    """
    After you train your network use its parameters to predict labels

    Returns:
    - y_prediction: array which shows predicted lables
    """
    y_prediction = None

    ###########################################################################
    # Implement this function. thats VERY easy to do
    ###########################################################################
    scores = self.loss(X)
    y_prediction = np.argmax(scores, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_prediction


