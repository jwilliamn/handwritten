#!/usr/bin/env python
# coding: utf-8

"""
    Modeling.modelD2
    ============================

    Handwritten digit recognition model
	Arquitecture: Conv. layer 6x6x1 > 6 stride 1
				  Conv. layer 5x5x6 > 12 stride 2
				  Conv. layer 4x4x12 > 24 stride 2
				  Fully Connected layer (relu + dropout)
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

import tensorflowvisu
import math
tf.set_random_seed(0)

# Global settings ####
# Path to save model parameters
model_path = '/home/williamn/Repository/handwritten/modeling'

# Hyperparameters
patch_size = 5 # F = 5 Size of filters
P1 = 6  # First convolutional patch size
P2 = 5  # Second convolutional patch size
P3 = 4  # Third convolutional patch size
S = 1 # stride
P = 0 # amount of zero padding

# Learning parameters
learning_rate_ = 0.001
adamLr = 1e-4
batch_size = 100
n_epochs = 3

# Network params
input_size = 32
n_channels = 1 # greyscale - Input image number of channels
D1 = 6  # First convolutional layer output depth (Number of filters)
D2 = 12  # Second convolutional layer output depth
D3 = 24  # Third convolutional layer output depth
n_hidden = 200 # fully connected neurons
n_classes = 10  # classes (0-9)



# Read in data ####
pickle_file = '/home/williamn/Repository/data/mnistAll/MNIST_digit_32x32.pickle'

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
#print(train_dataset[ran_image])
#a = train_dataset[ran_image].reshape([1, 1024])
#print(a[:,1:700])
label = train_labels[ran_image]
image = train_dataset[ran_image,:,:]
plt.title('Example: %d Label: %d' % (ran_image, label))
#####plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.imshow(image, cmap=plt.cm.gray)
plt.show()


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



# Placeholders for features and labels ####
X = tf.placeholder(tf.float32, [batch_size, input_size, input_size, n_channels], name='X_placeholder')
Y = tf.placeholder(tf.float32, [batch_size, n_classes], name='Y_placeholder')
X_val = tf.constant(valid_dataset)
X_test = tf.constant(test_dataset)

# Learning rate
lr = tf.placeholder(tf.float32)
# Dropout placeholder
keep_prob = tf.placeholder(tf.float32)


# Weights and bias ####
weights = {
	'w1' : tf.Variable(tf.truncated_normal(shape=[P1, P1, n_channels, D1], stddev=0.1)),
	'w2' : tf.Variable(tf.truncated_normal(shape=[P2, P2, D1, D2], stddev=0.1)),
	'w3' : tf.Variable(tf.truncated_normal(shape=[P3, P3, D2, D3], stddev=0.1)),
	'w4' : tf.Variable(tf.truncated_normal(shape=[input_size // 4 * input_size // 4 * D3, n_hidden], stddev=0.1)),
	'w5' : tf.Variable(tf.truncated_normal(shape=[n_hidden, n_classes], stddev=0.1))
}

biases = {
	'b1' : tf.Variable(tf.zeros([D1])),
	#'b1' : tf.Variable(tf.constant(0.1, tf.float32, shape=[D1])),
	'b2' : tf.Variable(tf.constant(0.1, tf.float32, shape=[D2])),
	'b3' : tf.Variable(tf.constant(0.1, tf.float32, shape=[D3])),
	'b4' : tf.Variable(tf.constant(0.1, tf.float32, shape=[n_hidden])),
	'b5' : tf.Variable(tf.constant(0.1, tf.float32, shape=[n_classes]))
}


# The model ####
def convnet_model(X, weights, biases, keep_prob):
	# 1st Convolutional layer
	conv1 = tf.nn.conv2d(X, weights['w1'], [1, 1, 1, 1], padding='SAME')
	h_conv1 = tf.nn.relu(conv1 + biases['b1'])
	
	# 2nd Convolutional layer
	conv2 = tf.nn.conv2d(h_conv1, weights['w2'], [1, 2, 2, 1], padding='SAME')
	h_conv2 = tf.nn.relu(conv2 + biases['b2'])
	
	# 3rd Convolutional layer
	conv3 = tf.nn.conv2d(h_conv2, weights['w3'], [1, 2, 2, 1], padding='SAME')
	h_conv3 = tf.nn.relu(conv3 + biases['b3'])

	# Fully connected layer
	shape = h_conv3.get_shape().as_list()
	h_conv3_flat = tf.reshape(h_conv3, [shape[0], shape[1]*shape[2]*shape[3]])
	h_fllc1 = tf.nn.relu(tf.matmul(h_conv3_flat, weights['w4']) + biases['b4'])

	# Dropout
	h_fllc1_drop = tf.nn.dropout(h_fllc1, keep_prob)

	# ReadOut layer
	o_layer = tf.matmul(h_fllc1_drop, weights['w5']) + biases['b5']
	return o_layer

logits = convnet_model(X, weights, biases, keep_prob)

train_pred = tf.nn.softmax(logits)

# Loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
loss = tf.reduce_mean(entropy)*100


# Training op
# using gradient descent to minimize loss
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
trainStepOpt = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)


# Accuracy function
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def accuracy2(predictions, labels):
	correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
	accuracy2 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy2

# accuracy of the trained model, between 0 (worst) and 1 (best)
#correct_prediction = tf.equal(tf.argmax(train_pred, 1), tf.argmax(Y, 1))
#accuracy3 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Learning rate decay
def learning_rate(step):
	max_lr = 0.003
	min_lr = 0.0001
	decay_speed = 2000.0
	return min_lr + (max_lr - min_lr)*math.exp(-step/decay_speed)


# Prediction ####
valid_pred = tf.nn.softmax(convnet_model(X_val, weights, biases, keep_prob = 1.0))
test_pred = tf.nn.softmax(convnet_model(X_test, weights, biases, keep_prob = 1.0))

# Visualize learning parameters
with tf.name_scope("Summaries"):
	#tf.summary.histogram("weights", [weights['w1'], weights['w2']])
	tf.summary.histogram("weights", weights['w1'])
	tf.summary.histogram('biases', biases['b1'])
	tf.summary.image('input', X, 3)
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('Accu', accuracy2(train_pred, Y))

	summary_op = tf.summary.merge_all()



with tf.Session() as sess:
	# to visualize using TensorBoard
	writer = tf.summary.FileWriter('./graphs/convnet_digits', sess.graph)

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
	
			lr_ = learning_rate(j)
			# TODO: run optimizer + fetch loss_batch
			_, loss_batch, preds, summary = sess.run([trainStepOpt, loss, train_pred, summary_op], feed_dict={X:X_batch, Y:Y_batch, keep_prob:0.75, lr:lr_})
	
			# Write logs at every iteration
			writer.add_summary(summary, (i*n_batches+j))
	
			total_loss += loss_batch
			if(j % 1000 == 0):
				print('		Minibatch loss at step %d: %f' % (j, loss_batch))
				print('		Minibatch accuracy: %.1f%%' % accuracy(preds, Y_batch))
				print('		Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(), valid_labels))
	
		print('	Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
	
	print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels))
	print('Total time: {0} seconds'.format(time.time() - start_time))
	
	print('Optimization Finished!')
	####print('Parameters: ', sess.run(weights), sess.run(biases))

	trained_weights = sess.run(weights)
	trained_biases = sess.run(biases)

	# Compute for visualization
	#def training_step(i, update_test_data, update_train_data):
	#	# training on batches of 100 images with 100 labels
	#	for j in range(n_batches):
	#		offset = (j * batch_size) % (train_labels.shape[0] - batch_size)
	#		X_batch = train_dataset[offset:(offset + batch_size),:,:,:]
	#		Y_batch = train_labels[offset:(offset + batch_size),:]
	#
	#		# learning rate decay
	#		max_learning_rate = 0.003
	#		min_learning_rate = 0.0001
	#		decay_speed = 2000.0
	#		learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-j/decay_speed)
	#
	#		# compute training values for visualisation
	#		if update_train_data:
	#			a, c, im, w, b = sess.run([accuracy3, loss, I, allweights, allbiases], {X: X_batch, Y: Y_batch, keep_prob: 1.0})
	#			print(str(j) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")
	#			datavis.append_training_curves_data(j, a, c)
	#			datavis.update_image1(im)
	#			datavis.append_data_histograms(j, w, b)
	#
	#		# compute test values for visualisation
	#		if update_test_data:
	#			a, c, im = sess.run([accuracy3, loss, It], {X: valid_dataset, Y: valid_labels, keep_prob: 1.0})
	#			print(str(j) + ": ********* epoch " + '_' + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
	#			datavis.append_test_curves_data(j, a, c)
	#			datavis.update_image2(im)
	#
	#		# the backpropagation training step
	#		sess.run(trainStepOpt, {X: X_batch, Y: Y_batch, lr: learning_rate, keep_prob: 0.75})
	#
	#datavis.animate(training_step, 10001, train_data_update_freq=20, test_data_update_freq=100)


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
	pickle_file = os.path.join(model_path, 'modelD2_param.pickle')

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
