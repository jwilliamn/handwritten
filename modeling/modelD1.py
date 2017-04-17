#!/usr/bin/env python
# coding: utf-8

"""
    Modeling.modelD1
    ============================

    Handwritten digit recognition model
	Arquitecture: 2 convolutional layers + maxpooling + 1 fc neural network
	Optimizer used: ADAM

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


# Hyperparameters
depth = 32 # K = 32 number of filters
patch_size = 5 # F = 5 Size of filters
S = 1 # stride
P = 0 # amount of zero padding

# Learning parameters
learning_rate = 0.001
adamLr = 1e-4
batch_size = 30 # 50
n_epochs = 3

# Network params
input_size = 32
n_channels = 1 # greyscale
n_hidden = 1024 # 1024 # fully connected neurons
n_classes = 10  # classes (0-9)


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
	dataset = dataset.reshape((-1, input_size, input_size, n_channels)).astype(np.float32)
	labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set_', train_dataset.shape, train_labels.shape, type(train_dataset))
print('Validation set_', valid_dataset.shape, valid_labels.shape, type(valid_dataset))
print('Test set_', test_dataset.shape, test_labels.shape, type(test_dataset))


# Visualize a random image (from original data)
#ran_image = ran.randint(0, train_dataset.shape[0])
#print('Random label: ' + str(train_labels[ran_image]))
#print('Random image: ')
#print(train_dataset[ran_image])
#label = train_labels[ran_image].argmax(axis=0)
#image = train_dataset[ran_image].reshape([32,32])
#plt.title('Example: %d Label: %d' % (ran_image, label))
#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#plt.imshow(image, cmap=plt.cm.gray)
#print('Real value' , '->', train_labels[ran_image].argmax(axis=0))
#plt.show()



# Step 2: Create placeholders for features and labels ####
X = tf.placeholder(tf.float32, [batch_size, input_size, input_size, n_channels], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, n_classes], name='Y_placeholder')
X_val = tf.constant(valid_dataset)
X_test = tf.constant(test_dataset)

# Dropout placeholder
keep_prob = tf.placeholder(tf.float32)


# Step 3: create weights and bias ####
weights = {
	'w1' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, n_channels, depth], stddev=0.1)),
	'w2' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, depth, 2*depth], stddev=0.1)),
	'w3' : tf.Variable(tf.random_normal(shape=[input_size // 4 * input_size // 4 * (2*depth), n_hidden], stddev=0.1)),
	'w4' : tf.Variable(tf.random_normal(shape=[n_hidden, n_classes], stddev=0.1))
}

biases = {
	#'b1' : tf.Variable(tf.zeros([depth])),
	'b1' : tf.Variable(tf.constant(0.1, shape=[depth])),
	'b2' : tf.Variable(tf.constant(0.1, shape=[2*depth])),
	'b3' : tf.Variable(tf.constant(0.1, shape=[n_hidden])),
	'b4' : tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


# Step 4: build model ####
def convnet_model(X, weights, biases, keep_prob):
#def convnet_model(X):
	# 1st Convolution layer
	conv1 = tf.nn.conv2d(X, weights['w1'], [1, 1, 1, 1], padding='SAME')
	h_conv1 = tf.nn.relu(conv1 + biases['b1'])
	h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

	# 2nd Convolution layer
	conv2 = tf.nn.conv2d(h_pool1, weights['w2'], [1, 1, 1, 1], padding='SAME')
	h_conv2 = tf.nn.relu(conv2 + biases['b2'])
	h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

	# Fully connected layer
	shape = h_pool2.get_shape().as_list()
	h_pool2_flat = tf.reshape(h_pool2, [shape[0], shape[1]*shape[2]*shape[3]])
	h_fllc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['w3']) + biases['b3'])

	# Dropout
	h_fllc1_drop = tf.nn.dropout(h_fllc1, keep_prob)

	# ReadOut layer
	o_layer = tf.matmul(h_fllc1_drop, weights['w4']) + biases['b4']
	return o_layer

logits = convnet_model(X, weights, biases, keep_prob)
#logits = convnet_model(X)


# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
loss = tf.reduce_mean(entropy)


# Step 6: define training op
# using gradient descent to minimize loss
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
trainStepOpt = tf.train.AdamOptimizer(learning_rate = adamLr).minimize(loss)


# Accuracy function
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def accuracy2(predictions, labels):
	correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy


# Step 7: Prediction*
train_pred = tf.nn.softmax(logits)
#valid_pred = tf.nn.softmax(convnet_model(X_val))
#test_pred = tf.nn.softmax(convnet_model(X_test))
valid_pred = tf.nn.softmax(convnet_model(X_val, weights, biases, keep_prob = 1.0))
test_pred = tf.nn.softmax(convnet_model(X_test, weights, biases, keep_prob = 1.0))

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
			_, loss_batch, preds = sess.run([trainStepOpt, loss, train_pred], feed_dict={X:X_batch, Y:Y_batch, keep_prob:0.5})

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
		image = test_dataset[ran_image].reshape([32,32])     #reshape([32,32])
		plt.title('Example test: %d Label: %d' % (ran_image, label))
		plt.imshow(image, cmap=plt.cm.gray)
		plt.show()



	# Saving trained parameters the data for later reuse
	pickle_file = os.path.join(model_path, 'modelD1_param.pickle')

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
