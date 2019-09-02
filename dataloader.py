import numpy as np
from sklearn.feature_selection import VarianceThreshold

root = 'Fashion-MNIST/'
def select_features():
      # Loading Dataset
      train_data = np.loadtxt(root+'trainData.csv', dtype=np.float32, delimiter=',')
      train_labels = np.loadtxt(root+'/trainLabels.csv', dtype=np.int32, delimiter=',')
      test_data = np.loadtxt(root+'/testData.csv', dtype=np.float32, delimiter=',')
      test_labels = np.loadtxt(root+'/testLabels.csv', dtype=np.int32, delimiter=',')
      class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
     
      tr_samples_size,_ = train_data.shape
      tr_samples_size, feature_size = train_data.shape
      te_samples_size, _ = test_data.shape

      return train_data, train_labels, test_data, test_labels, class_names, tr_samples_size, te_samples_size, len(class_names), feature_size