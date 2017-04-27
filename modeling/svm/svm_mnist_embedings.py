print(__doc__)
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://machinethoughts.me
# License: BSD 3 clause


# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
from time import time

from mnist_helpers import *

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn.decomposition import PCA

import os

import matplotlib.pyplot as plt
import random as ran
import pickle


#fetch original mnist dataset
from sklearn.datasets import fetch_mldata



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


#full dataset classification
X_data =images/255.0
Y = targets

#split data to train and test 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

# Create a classifier: a support vector classifier
kernel_svm = svm.SVC(gamma=.2)
linear_svm = svm.LinearSVC()

# create pipeline from kernel approximation
# and linear svm
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1)

fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", svm.LinearSVC())])

nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("svm", svm.LinearSVC())])

# fit and predict using linear and kernel svm:

import datetime as dt
# We learn the digits on train part

kernel_svm_start_time = dt.datetime.now()
print ('Start kernel svm learning at {}'.format(str(kernel_svm_start_time)))
kernel_svm.fit(X_train, y_train)
kernel_svm_end_time = dt.datetime.now()
elapsed_time = kernel_svm_end_time - kernel_svm_start_time
print ('End kernel svm learning at {}'.format(str(kernel_svm_end_time)))
print ('Elapsed learning {}'.format(str(elapsed_time)))

kernel_svm_start_time = dt.datetime.now()
kernel_svm_score = kernel_svm.score(X_test, y_test)
elapsed_time = dt.datetime.now() - kernel_svm_start_time
print ('Prediction takes {}'.format(str(elapsed_time)))

linear_svm_time = dt.datetime.now()
linear_svm.fit(X_train, y_train)
linear_svm_score = linear_svm.score(X_test, y_test)
linear_svm_time = dt.datetime.now() - linear_svm_time


#aprox sample sizes, used for ploting 
sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
fourier_times = []
nystroem_times = []


for D in sample_sizes:
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm.fit(X_train, y_train)
    nystroem_times.append(time() - start)

    start = time()
    fourier_approx_svm.fit(X_train, y_train)
    fourier_times.append(time() - start)

    fourier_score = fourier_approx_svm.score(X_test, y_test)
    nystroem_score = nystroem_approx_svm.score(X_test, y_test)
    nystroem_scores.append(nystroem_score)
    fourier_scores.append(fourier_score)

# plot the results:
plt.figure(figsize=(8, 8))
accuracy = plt.subplot(211)
# second y axis for timeings
timescale = plt.subplot(212)

accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
timescale.plot(sample_sizes, nystroem_times, '--',
               label='Nystroem approx. kernel')

accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
timescale.plot(sample_sizes, fourier_times, '--',
               label='Fourier approx. kernel')

# horizontal lines for exact rbf and linear kernels:
accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [linear_svm_score, linear_svm_score], label="linear svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [linear_svm_time, linear_svm_time], '--', label='linear svm')

accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [kernel_svm_score, kernel_svm_score], label="rbf svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [kernel_svm_time, kernel_svm_time], '--', label='rbf svm')

# vertical line for dataset dimensionality = 64
accuracy.plot([64, 64], [0.7, 1], label="n_features")

# legends and labels
accuracy.set_title("Classification accuracy")
timescale.set_title("Training times")
accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
accuracy.set_xticks(())
accuracy.set_ylim(np.min(fourier_scores), 1)
timescale.set_xlabel("Sampling steps = transformed feature dimension")
accuracy.set_ylabel("Classification accuracy")
timescale.set_ylabel("Training time in seconds")
accuracy.legend(loc='best')
timescale.legend(loc='best')

# visualize the decision surface, projected down to the first
# two principal components of the dataset
pca = PCA(n_components=8).fit(data_train)

X = pca.transform(data_train)

# Gemerate grid along first two principal components
multiples = np.arange(-2, 2, 0.1)
# steps along first component
first = multiples[:, np.newaxis] * pca.components_[0, :]
# steps along second component
second = multiples[:, np.newaxis] * pca.components_[1, :]
# combine
grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
flat_grid = grid.reshape(-1, data.shape[1])

# title for the plots
titles = ['SVC with rbf kernel',
          'SVC (linear kernel)\n with Fourier rbf feature map\n'
          'n_components=100',
          'SVC (linear kernel)\n with Nystroem rbf feature map\n'
          'n_components=100']

plt.tight_layout()
plt.figure(figsize=(12, 5))

# predict and plot
for i, clf in enumerate((kernel_svm, nystroem_approx_svm,
                         fourier_approx_svm)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(1, 3, i + 1)
    Z = clf.predict(flat_grid)

    # Put the result into a color plot
    Z = Z.reshape(grid.shape[:-1])
    plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=targets_train, cmap=plt.cm.Paired)

    plt.title(titles[i])
plt.tight_layout()
plt.show()