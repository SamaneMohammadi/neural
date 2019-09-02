import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Loading Dataset
train_data = np.loadtxt('Fashion-MNIST-1/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('Fashion-MNIST-1/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('Fashion-MNIST-1/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('Fashion-MNIST-1/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']




# Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
sel = VarianceThreshold(threshold=0.86*(1-0.86))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
print('Train Data Samples:',tr_samples_size,
      ', Test Data Samples',te_samples_size,
      ', Feature Size(after feature-selection):', feature_size)