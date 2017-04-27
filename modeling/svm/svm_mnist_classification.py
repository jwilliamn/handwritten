print(__doc__)
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://ksopyla.com
# License: MIT

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata

import os

import matplotlib.pyplot as plt
import random as ran
import pickle


# import custom module
from mnist_helpers import *


#mnist = fetch_mldata('MNIST original', data_home='./')

#minist object contains: data, COL_NAMES, DESCR, target fields
#you can check it by running
#mnist.keys()


# Step 1: Read in data ####
#pickle_file = '/home/williamn/Repository/data/mnistAll/allMNIST.pickle' #-- Cambiar nombre apro
pickle_file = '/home/williamn/Repository/data/mnistAll/MNIST_digit_32x32.pickle' #-- Cambiar nombre apro

with open(pickle_file, 'rb') as f:
	allmnist = pickle.load(f)
	train_dataset = allmnist['train_dataset']
	train_labels = allmnist['train_labels']
	valid_dataset = allmnist['valid_dataset']
	valid_labels = allmnist['valid_labels']
	test_dataset = allmnist['test_dataset']
	test_labels = allmnist['test_labels']
	del allmnist  # free up memory

valid_dataset = valid_dataset[58000:68856,:,:]
valid_labels = valid_labels[58000:68856,]
test_dataset = test_dataset[48000:58646,:,:]
test_labels = test_labels[48000:58646,]
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# visualize a random image (resized from original)
ran_image = ran.randint(0, train_dataset.shape[0])
print('Random label: ' + str(train_labels[ran_image]))
print('Random image: ')
label = train_labels[ran_image]
image = train_dataset[ran_image].reshape([32,32])
plt.title('Example: %d Label: %d' % (ran_image, label))
plt.imshow(image, cmap=plt.cm.gray)
plt.show()


# Network params
input_size = 32
n_channels = 1 # greyscale
n_hidden = 1024 # 1024 # fully connected neurons
n_classes = 10  # classes (0-9)

# Reformat into a shape that's more adapted to the model
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, input_size*input_size)).astype(np.float32)
	#labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
	return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set_', train_dataset.shape, train_labels.shape, type(train_dataset))
print('Validation set_', valid_dataset.shape, valid_labels.shape, type(valid_dataset))
print('Test set_', test_dataset.shape, test_labels.shape, type(test_dataset))


# visualize a random image (resized from original)
ran_image = ran.randint(0, train_dataset.shape[0])
print('Random label: ' + str(train_labels[ran_image]))
print('Random image: ')
label = train_labels[ran_image].argmax(axis=0)
image = train_dataset[ran_image].reshape([32,32])
plt.title('Example: %d Label: %d' % (ran_image, label))
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
#images = mnist.data
#targets = mnist.target

images = train_dataset
targets = train_labels

# Let's have a look at the random 16 images, 
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array

#pick  random indexes from 0 to size of our dataset
show_some_digits(images,targets)

#---------------- classification begins -----------------
#scale data for [0,255] -> [0,1]
#sample smaller size for testing
#rand_idx = np.random.choice(images.shape[0],10000)
#X_data =images[rand_idx]/255.0
#Y      = targets[rand_idx]

#full dataset classification
X_data = images/255.0
Y = targets

#split data to train and test
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)


############### Classification with grid search ##############
# If you don't want to wait, comment this section and uncommnet section below with
# standalone SVM classifier

# Warning! It takes really long time to compute this about 2 days

# Create parameters grid for RBF kernel, we have to set C and gamma
from sklearn.model_selection import GridSearchCV

# generate matrix with all gammas
# [ [10^-4, 2*10^-4, 5*10^-4], 
#   [10^-3, 2*10^-3, 5*10^-3],
#   ......
#   [10^3, 2*10^3, 5*10^3] ]
#gamma_range = np.outer(np.logspace(-4, 3, 8),np.array([1,2, 5]))
gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
gamma_range = gamma_range.flatten()

# generate matrix with all C
#C_range = np.outer(np.logspace(-3, 3, 7),np.array([1,2, 5]))
C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
# flatten matrix, change to 1D numpy array
C_range = C_range.flatten()

parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}

svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2)


start_time = dt.datetime.now()
print('Start param searching at {}'.format(str(start_time)))

grid_clsf.fit(X_train, y_train)

elapsed_time= dt.datetime.now() - start_time
print('Elapsed time, param searching {}'.format(str(elapsed_time)))
sorted(grid_clsf.cv_results_.keys())

classifier = grid_clsf.best_estimator_
params = grid_clsf.best_params_



scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

plot_param_space_scores(scores, C_range, gamma_range)


######################### end grid section #############



################ Classifier with good params ###########
# Create a classifier: a support vector classifier

# param_C = 5
# param_gamma = 0.05
#classifier = svm.SVC(C=param_C,gamma=param_gamma)

# We learn the digits on train part
# start_time = dt.datetime.now()
# print('Start learning at {}'.format(str(start_time)))
# classifier.fit(X_train, y_train)
# end_time = dt.datetime.now() 
# print('Stop learning {}'.format(str(end_time)))
# elapsed_time= end_time - start_time
# print('Elapsed learning {}'.format(str(elapsed_time)))


########################################################




# Now predict the value of the test
expected = y_test
predicted = classifier.predict(X_test)

show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


