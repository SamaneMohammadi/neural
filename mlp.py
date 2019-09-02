# initialization

import numpy as np
import matplotlib.pyplot as plt

from neural_net import MLPNet

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# %load_ext autoreload
# %autoreload 2

def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))




# Create our network and toy dataset

features = 4#5
hidden_size = 8#6
classes = 3
inputs = 5
np.random.seed(0)

net = MLPNet(features, hidden_size, classes, std=1e-1)
X = 10 * np.random.randn(inputs, features)
y = np.array([1, 0, 0, 2, 1])



scores = net.loss(X)
print ('Your scores:\n',scores)
labels = np.argmax(scores, axis = 1)
print ('Predicted labels:\n',labels)



loss, _ = net.loss(X, y, reg=0.1)
print("your network loss is:",loss)



loss, grads = net.loss(X, y, reg=0.1)
print('grads of W1\n',grads['W1'])
print('grads of b1\n',grads['b1'])
print('grads of W2\n',grads['W2'])
print('grads of b2\n',grads['b2'])




net = MLPNet(features, hidden_size, classes, std=1e-1)
stats = net.train(X, y, X, y,
            alpha=1e-1, reg=1e-5,
            num_iters=100)
print(net.predict(X))
print ('Final training loss: ', stats['loss_train'][-1])

# plot the loss history
plt.plot(stats['loss_train'])#'loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()





from dataloader import select_features#from FeatureSelection import select_features
import numpy as np
import os
from scipy.misc import imread
#import cv2
import matplotlib.pyplot as plt
import matplotlib
val_num = 1000
train_num = 49000
test_num = 10000
train_data, train_labels, test_data, test_labels,\
    class_names, n_train, n_test, n_class, n_features = select_features()


# Subsample the data
mask = range(train_num, n_train)
X_val = train_data[mask]
y_val = train_labels[mask]
mask = range(train_num)
X_train = train_data[mask]
y_train = train_labels[mask]
mask = range(test_num)
X_test = test_data[mask]
y_test = test_labels[mask]

# # Normalize the data: subtract the mean image
# mean_image = np.mean(X_train, axis=0)
# X_train -= mean_image
# X_val -= mean_image
# X_test -= mean_image

print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Validation data shape: ', X_val.shape)
print ('Validation labels shape: ', y_val.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)




input_size = n_features
print("input size:", input_size)
hidden_size = 500
num_classes = 10
net = MLPNet(input_size, hidden_size, num_classes,std=1e-2)

# Train the network
# وقتی نرخ را برابر یک ده هزارم گذاشتیم بماند اصلا همگرا نمی شد، با زیاد کردن نرخ به یک صدم در ایپاک های اولیه همگرایی شروع شد
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=100, batch_size=200,
            alpha=1e-2, alpha_decay=0.95,
            reg=0.5, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print ('Validation accuracy: ', val_acc)


# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_train'])
plt.title('Training Loss') # Loss')
plt.xlabel('Itteration')
plt.ylabel('Loss')
plt.show() # اضافه شده
print() # اضافه شده

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc'], label='train')
plt.title('Training Accuracy') # اضافه شده
plt.xlabel('Epoch') # اضافه شده
plt.ylabel('Accuracy') # اضافه شده
plt.show() # اضافه شده
print() # اضافه شده

plt.plot(stats['va_acc'], label='val') # ['val_acc']
plt.title('Classification accuracy ')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()



best_net = None  # store the best model into this
#################################################################################
# Tune hyperparameters with validation set and store your best model in best_net#
#################################################################################
best_val_acc = 0
for hidden_size in [50, 100, 200, 300, 500]:
    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        for reg in [0.1, 0.3, 0.5, 0.7]:
            print('\nhidden_size: %d, learning_rate: %f, regularization_weight: %f' % (hidden_size, alpha, reg))
            net = MLPNet(n_features, hidden_size, num_classes, std=1e-2)
            stats = net.train(X_train, y_train, X_val, y_val,
                              num_iters=10, batch_size=200,
                              alpha=alpha, alpha_decay=0.95,
                              reg=reg)
            val_acc = (net.predict(X_val) == y_val).mean()
            print('Validation accuracy: ', val_acc)

            if (val_acc > best_val_acc):
                best_val_acc = val_acc
                best_net = net
#################################################################################
#                                END OF YOUR CODE                               #
#################################################################################






import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
        'hidden_layer_sizes': [(100,), (200,),(400,)],
        'learning_rate_init': [0.1, 0.01,0.0001],
        'alpha': [0.1, 0.3, 0.5] #L2 penalty (regularization term) parameter.
    }
estimator = GridSearchCV(
        MLPClassifier(batch_size = 200, max_iter=10, learning_rate = 'invscaling', solver = 'sgd', random_state = 2, verbose=False),
        param_grid = param_grid, verbose=2)
estimator.fit(X_train, y_train)

clf = estimator.best_estimator_

print("Training accuracy: %f" % clf.score(X_train, y_train))
print("Validation accuracy: %f" % clf.score(X_val, y_val))







from visualization import visualize
print(clf) # چاپ پارامترهای بهترین کلاسیفایر یافت شده
weight=clf.coefs_[0]
weight=np.asarray(weight)
def show_net_weights(W1):
  W1 = W1[:,0:50].reshape(28, 28, 1, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

show_net_weights(weight)



from sklearn.neural_network import MLPClassifier
from visualization import visualize

clf_best = MLPClassifier(hidden_layer_sizes=(400,), learning_rate_init=0.1, alpha=0.1,
                    solver='sgd', batch_size = 200, max_iter=40, random_state=2
                    )
clf_best.fit(X_train, y_train)

weight=clf_best.coefs_[0]
weight=np.asarray(weight)
def show_net_weights(W1):
  W1 = W1[:,0:50].reshape(28, 28, 1, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

show_net_weights(weight)



print("Test accuracy: %f" % clf.score(X_test, y_test))


'''  '''
def Normalize(X):
    N = X.shape[0]
    mu = np.sum(X, axis = 0) / N
    X_normalized = X - mu
    sigma2 = np.sum(np.square(X_normalized), axis = 0) / N
    sigma = np.sqrt(sigma2)
    sigma += 0.0001 #epsilon برای جلوگیری از تقسیم بر صفر شدن
    X_normalized /= sigma
    return X_normalized

X_train_normalized = Normalize(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(400,), learning_rate_init=0.1, alpha=0.1,
                    solver='sgd', batch_size = 200, max_iter=10, random_state=2, verbose=True
                    )
mlp.fit(X_train_normalized, y_train)

#print(np.min(X_train))
#print(np.max(X_train))
print("Training accuracy: %f" % mlp.score(X_train_normalized, y_train))
X_val_normalized = Normalize(X_val)
print("Validation accuracy: %f" % mlp.score(X_val_normalized, y_val))


X_test_normalized = Normalize(X_test)
print("Test accuracy: %f" % mlp.score(X_test_normalized, y_test))