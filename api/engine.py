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


# Prediction ####
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
	
