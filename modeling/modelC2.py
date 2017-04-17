""" 

Handwritten characters recognition model
Arquitecture: Lenet 5
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
# Path to save model parameters
model_path = '/home/williamn/Repository/hand_Written_Character_Recognition'

# Hyperparameters
depth = 16 # K = 6 number of filters
patch_size = 5 # F = 5 Size of filters
S = 1 # stride
P = 0 # amount of zero padding

# Learning parameters
learning_rate = 0.001  # 0.001
adamLr = 1e-4
batch_size = 16  # 16
n_epochs = 2

# Network params
input_size = 32
n_channels = 1 # greyscale
n_classes = 26  # classes (A-Z)


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

# Reformat into a shape that's more adapted to the convnet model
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, input_size, input_size, n_channels)).astype(np.float32)
	labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


# Step 2: Create placeholders for features and labels ####
X = tf.placeholder(tf.float32, [batch_size, input_size, input_size, n_channels], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, n_classes], name='Y_placeholder')
X_val = tf.constant(valid_dataset)
X_test = tf.constant(test_dataset)


# Step 3: create weights and bias ####
weights = {
	'w1' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, n_channels, depth], stddev=0.1)),
	'w2' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, depth, 2*depth], stddev=0.1)),
	'w3' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, 2*depth, 120], stddev=0.1)),
	'w4' : tf.Variable(tf.random_normal(shape=[input_size // 4 * input_size // 4 *120, 84], stddev=0.1)),
	'w5' : tf.Variable(tf.random_normal(shape=[84, n_classes], stddev=0.1))
}

biases = {
	'b1' : tf.Variable(tf.zeros([depth])),
	'b2' : tf.Variable(tf.constant(1.0, shape=[2*depth])),
	'b3' : tf.Variable(tf.constant(1.0, shape=[120])),
	'b4' : tf.Variable(tf.constant(1.0, shape=[84])),
	'b5' : tf.Variable(tf.constant(1.0, shape=[n_classes]))
}


# Step 4: build model ####
def lenet_model(X):
	# 1st Convolution layer
	conv = tf.nn.conv2d(X, weights['w1'], [1, 1, 1, 1], padding='SAME')
	hidden = tf.nn.relu(conv + biases['b1'])
	hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

	# 2nd Convolution layer
	conv = tf.nn.conv2d(hidden, weights['w2'], [1, 1, 1, 1], padding='SAME')
	hidden = tf.nn.relu(conv + biases['b2'])
	hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

	# 3rd Convolution layer
	conv = tf.nn.conv2d(hidden, weights['w3'], [1, 1, 1, 1], padding='SAME')
	hidden = tf.nn.relu(conv + biases['b3'])
	#hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')

	# Fully connected layer
	shape = hidden.get_shape().as_list()
	reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
	print('shape', shape)
	print('reshape', reshape)
	print('weights4', weights['w4'])
	hidden = tf.nn.relu(tf.matmul(reshape, weights['w4']) + biases['b4'])

	# Output
	o_layer = tf.matmul(hidden, weights['w5']) + biases['b5']
	return o_layer

logits = lenet_model(X)


# Step 5: define loss function ####
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)


# Step 6: define training op
# using gradient descent to minimize loss
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
trainStepOpt = tf.train.AdamOptimizer(learning_rate = adamLr).minimize(loss)



# Accuracy function
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])



# Step 7: Prediction*
train_pred = tf.nn.softmax(logits)
valid_pred = tf.nn.softmax(lenet_model(X_val))
test_pred = tf.nn.softmax(lenet_model(X_test))


with tf.Session() as sess:

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(train_dataset.shape[0]/batch_size)

	print('Initialized!', n_batches)

	for i in range(n_epochs):
		total_loss = 0

		for j in range(n_batches):
			offset = (j * batch_size) % (train_labels.shape[0] - batch_size)
			X_batch = train_dataset[offset:(offset + batch_size),:,:,:]
			Y_batch = train_labels[offset:(offset + batch_size),:]

			_, loss_batch, preds = sess.run([trainStepOpt, loss, train_pred], feed_dict={X:X_batch, Y:Y_batch})

			total_loss += loss_batch
			if(j % 1000 == 0):
				print('		Minibatch loss at step %d: %f' % (j, loss_batch))
				print('		Minibatch accuracy: %.1f%%' % accuracy(preds, Y_batch))
				print('		Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(), valid_labels))
		print('	Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels))
	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!')


	trained_weights = sess.run(weights)
	trained_biases = sess.run(biases)

	# Saving trained parameters for later reuse
	pickle_file = os.path.join(model_path, 'modelC2_param.pickle')

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
