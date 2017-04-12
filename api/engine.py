""" 

Handwritten characters recognition implementation on real data
Arquitecture: 2 convolutional layers + maxpooling + fc neural network
Optimizer used: Stochastic gradient descent 
Author: J.W.

"""

import tensorflow as tf
import numpy as np
import time
import os

import matplotlib.pyplot as plt
import random as ran
import pickle


# Global settings ####
# Path to saved model parameters
model_path = '/home/williamn/Repository/handwritten/'

# Parameters model
learning_rate = 0.05
batch_size = 16  # 100
patch_size = 5
n_epochs = 2 # 15

# Network Parameters
image_size = 32
n_input = 1024  # input nodes
depth = 16
n_channels = 1 # grayscale
n_hidden = 64  # hidden layer nodes
n_classes = 26  # total classes (A - Z)


# Step 1: Read in params ####
def getModelParams(param_file):
	with open(param_file, 'rb') as f:
		param = pickle.load(f)
		weights = param['weights']
		biases = param['biases']
		del param  # free up memory

	return weights, biases


# Reformat into a shape that's more adapted to the model
def reformat(dataset, labels=None):
	dataset = dataset.reshape((-1, image_size, image_size, n_channels)).astype(np.float32)
	#labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
	return dataset, labels






# Step 4: trained model ####
def convnet_model(X, weights, biases):
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
	reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
	hidden = tf.nn.relu(tf.matmul(reshape, weights['w3']) + biases['b3'])

	# Output
	o_layer = tf.matmul(hidden, weights['w4']) + biases['b4']
	return o_layer

def convnet_model_d(X, weights, biases, keep_prob):
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


# Prediction of characters ####
def predictImage(image_data):
	print('Real image', image_data.shape, type(image_data))

	# Read model parameters ##
	#param_file = os.path.join(model_path, 'allMNIST_param.pickle')
	param_file = os.path.join(model_path, 'modeling/modelC1_param.pickle')
	
	weights, biases = getModelParams(param_file)

	# Reformat image to convnet shape
	image_dataset, _ = reformat(image_data, labels=None)

	# Define data constant ####
	X = tf.constant(image_dataset)

	# Pred run
	image_pred = tf.nn.softmax(convnet_model(X, weights, biases))

	with tf.Session() as sess:
		start_time = time.time()
		sess.run(tf.global_variables_initializer())	

		#print('Prediction initialized!')

		preds_ = image_pred.eval()
		pred_label = tf.argmax(preds_, 1)
		pred_label_ = sess.run(pred_label)
		#print('Predicted value = %d' % sess.run(pred_label))
		#print(sess.run(pred_label), '->', chr(sess.run(pred_label)+ord('A')))
		

		#print('Total time: {0} seconds'.format(time.time() - start_time))
		#print('Testing Finished!')
		
		#image = image_data.reshape([32,32])
		#plt.title('Image - FSU')
		#plt.imshow(image, cmap=plt.cm.gray)
		#plt.show()

	return pred_label_
	

# Prediction of digits ####
def predictImage_dig(image_data):
	print('Real digit image', image_data.shape, type(image_data))

	# Read model parameters ##
	#param_file = os.path.join(model_path, 'allMNIST_param.pickle')
	param_file = os.path.join(model_path, 'modeling/modelD1_param_.pickle')
	
	weights, biases = getModelParams(param_file)

	# Reformat image to convnet shape
	image_dataset, _ = reformat(image_data, labels=None)

	# Define data constant ####
	X = tf.constant(image_dataset)

	# Pred run
	image_pred = tf.nn.softmax(convnet_model_d(X, weights, biases, keep_prob=1.0))

	with tf.Session() as sess:
		start_time = time.time()
		sess.run(tf.global_variables_initializer())	

		#print('Prediction initialized!')

		preds_ = image_pred.eval()
		pred_label = tf.argmax(preds_, 1)
		pred_label_ = sess.run(pred_label)
		#print('Predicted value = %d' % sess.run(pred_label))
		#print(sess.run(pred_label), '->', chr(sess.run(pred_label)+ord('A')))
		

		#print('Total time: {0} seconds'.format(time.time() - start_time))
		#print('Testing Finished!')
		
		#image = image_data.reshape([32,32])
		#plt.title('Image - FSU')
		#plt.imshow(image, cmap=plt.cm.gray)
		#plt.show()

	return pred_label_
	
