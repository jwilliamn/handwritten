#!/usr/bin/env python
# coding: utf-8

"""
    Modeling.modelC1
    ============================

    Handwritten characters recognition model
	Arquitecture: 2 convolutional layers + maxpooling + fc neural network
	Optimizer used: Stochastic gradient descent 

    _copyright_ = 'Copyright (c) 2017 J.W.', see AUTHORS for more details
    _license_ = GNU General Public License, see LICENSE for more details
"""

import tensorflow as tf
import numpy as np
import time
import os

import matplotlib.pyplot as plt
import random as ran
import pickle


# Global settings ####
# Path to save model parameters
#model_path = '/home/williamn/Repository/hand_Written_Character_Recognition'
model_path = '/home/williamn/Repository/handwritten/modeling'

# Parameters model
learning_rate =  0.05 # 0.05
batch_size = 16  # 100
patch_size = 5
n_epochs = 3 # 15

# Network Parameters
image_size = 32
n_input = 1024  # input nodes
depth = 16
n_channels = 1 # grayscale
n_hidden = 64  # hidden layer nodes
n_classes = 26  # total classes (A - Z)


# Step 1: Read in data ####
#pickle_file = '/home/williamn/Repository/data/mnistAll/allMNIST.pickle' #-- Cambiar nombre apro
pickle_file = '/home/williamn/Repository/data/mnistAll/MNIST_32x32.pickle' #-- Cambiar nombre apro

with open(pickle_file, 'rb') as f:
	allmnist = pickle.load(f)
	train_dataset = allmnist['train_dataset']
	train_labels = allmnist['train_labels']
	valid_dataset = allmnist['valid_dataset']
	valid_labels = allmnist['valid_labels']
	test_dataset = allmnist['test_dataset']
	test_labels = allmnist['test_labels']
	del allmnist  # free up memory

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# visualize a random image (resized from original)
#ran_image = ran.randint(0, train_dataset.shape[0])
#print('Random label: ' + str(train_labels[ran_image]))
#print('Random image: ')
#print(train_dataset[ran_image])
#a = train_dataset[ran_image].reshape([1, 1024])
#print(a[:,1:700])
#label = train_labels[ran_image]
#image = train_dataset[ran_image,:,:]
#plt.title('Example: %d Label: %d' % (ran_image, label))
#####plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#plt.imshow(image, cmap=plt.cm.gray)
#plt.show()


# Reformat into a shape that's more adapted to the model
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, n_channels)).astype(np.float32)
	labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set_', train_dataset.shape, train_labels.shape, type(train_dataset))
print('Validation set_', valid_dataset.shape, valid_labels.shape, type(valid_dataset))
print('Test set_', test_dataset.shape, test_labels.shape, type(test_dataset))

# Visualize a random image (from original data)
ran_image = ran.randint(0, train_dataset.shape[0])
print('Random label: ' + str(train_labels[ran_image]))
print('Random image: ')
#print(train_dataset[ran_image])
label = train_labels[ran_image].argmax(axis=0)
image = train_dataset[ran_image].reshape([32,32])
plt.title('Example: %d Label: %d' % (ran_image, label))
#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.imshow(image, cmap=plt.cm.gray)
print(train_labels[ran_image].argmax(axis=0), '->', chr(train_labels[ran_image].argmax(axis=0)+ord('A')))
plt.show()


# Step 2: Create placeholders for features and labels ####
# each image in the allMNIST data is of shape 32*32 = 1024
# therefore, each image is represented with a 1x1024 tensor
# there are 26 classes for each image, corresponding to English letters A - Z. 
X = tf.placeholder(tf.float32, [batch_size, image_size, image_size, n_channels], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, n_classes], name='Y_placeholder')
X_val = tf.constant(valid_dataset)
X_test = tf.constant(test_dataset)


# Step 3: create weights and bias ####
# weights and biases are random initialized
# shape of w depends on the dimension of X and the next layer (i.e so that Y = X * w + b)
# shape of b depends on the next layers (i.e Y)
weights = {
	'w1' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, n_channels, depth], stddev=0.1)),
	'w2' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, depth, depth], stddev=0.1)),
	'w3' : tf.Variable(tf.random_normal(shape=[image_size // 4 * image_size // 4 * depth, n_hidden], stddev=0.1)),
	'w4' : tf.Variable(tf.random_normal(shape=[n_hidden, n_classes], stddev=0.1))
}

biases = {
	'b1' : tf.Variable(tf.zeros([depth])),
	'b2' : tf.Variable(tf.constant(1.0, shape=[depth])),
	'b3' : tf.Variable(tf.constant(1.0, shape=[n_hidden])),
	'b4' : tf.Variable(tf.constant(1.0, shape=[n_classes]))
}


# Step 4: build model ####
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
def convnet_model(X):
	# 1st Convolution layer
	conv = tf.nn.conv2d(X, weights['w1'], [1, 1, 1, 1], padding='SAME')
	hidden = tf.nn.relu(conv + biases['b1'])
	hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

	# 2nd Convolution layer
	conv = tf.nn.conv2d(hidden, weights['w2'], [1, 1, 1, 1], padding='SAME')
	hidden = tf.nn.relu(conv + biases['b2'])
	hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

	# Fully connected layer
	shape = hidden.get_shape().as_list()
	print('shape 1',shape)
	reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
	print('reshape 2', reshape)
	print('w3', weights['w3'])
	hidden = tf.nn.relu(tf.matmul(reshape, weights['w3']) + biases['b3'])

	# Output
	o_layer = tf.matmul(hidden, weights['w4']) + biases['b4']
	return o_layer

logits = convnet_model(X)

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)


# Step 6: define training op
# using gradient descent to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# Accuracy function
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


# Step 7: Prediction*
train_pred = tf.nn.softmax(logits)
valid_pred = tf.nn.softmax(convnet_model(X_val))
test_pred = tf.nn.softmax(convnet_model(X_test))

with tf.Session() as sess:
	# to visualize using TensorBoard
	#writer = tf.summary.FileWriter('./graphs/convnet', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(train_dataset.shape[0]/batch_size)

	print('Initialized!', n_batches)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for j in range(n_batches):
			offset = (j * batch_size) % (train_labels.shape[0] - batch_size)
			X_batch = train_dataset[offset:(offset + batch_size),:,:,:]
			Y_batch = train_labels[offset:(offset + batch_size),:]
			# TO-DO: run optimizer + fetch loss_batch
			_, loss_batch, preds = sess.run([optimizer, loss, train_pred], feed_dict={X:X_batch, Y:Y_batch})

			total_loss += loss_batch
			if(j % 1000 == 0):
				print('		Minibatch loss at step %d: %f' % (j, loss_batch))
				print('		Minibatch accuracy: %.1f%%' % accuracy(preds, Y_batch))
				print('		Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(), valid_labels))
		print('	Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels))
	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	#print('Parameters: ', sess.run(weights), sess.run(biases))

	trained_weights = sess.run(weights)
	trained_biases = sess.run(biases)

	# Visualization of the predicctions on test data
	for _ in range(10):
		ran_image = ran.randint(0, test_dataset.shape[0])
		print('Random test label {0}'.format(test_labels[ran_image]))
		
		preds_ = test_pred.eval()
		pred_label = tf.argmax(preds_, 1)
		print('Predicted value = %d' % sess.run(pred_label[ran_image]))

		label = test_labels[ran_image].argmax(axis=0)
		#image = test_dataset[ran_image].reshape([32,32])       #reshape([32,32])  
		#plt.title('Example test: %d Label: %d' % (ran_image, label))
		#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
		#plt.show()

		image = test_dataset[ran_image].reshape([32,32])     #reshape([32,32])
		plt.title('Example test: %d Label: %d' % (ran_image, label))
		plt.imshow(image, cmap=plt.cm.gray)
		plt.show()



	# Saving trained parameters the data for later reuse
	pickle_file = os.path.join(model_path, 'modelC1_param.pickle')

	try:
		f = open(pickle_file, 'wb')
		save = {
			'weights': trained_weights,
			'biases': trained_biases,
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data to', pickle_file, ':', e)
		raise

	statinfo = os.stat(pickle_file)
	print('Compressed pickle size:', statinfo.st_size)


	#writer.close()
