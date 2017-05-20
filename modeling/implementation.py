""" 

Handwritten characters recognition implementation
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
# Path to save model parameters
model_path = '/home/williamn/Repository/handwritten'

# Parameters model
#learning_rate = 0.05
#batch_size = 16  # 100
#patch_size = 5
#n_epochs = 2 # 15

# Network Parameters
image_size = 32
#n_input = 1024  # input nodes
#depth = 16
n_channels = 1 # grayscale
#n_hidden = 64  # hidden layer nodes
n_classes = 26  # total classes (A - Z)


# Step 1: Read in data ####
pickle_file = '/home/williamn/Repository/data/mnistAll/MNIST_32x32.pickle' #-- Cambiar nombre apro
with open(pickle_file, 'rb') as f:
	allmnist = pickle.load(f)
	#train_dataset = allmnist['train_dataset']
	#train_labels = allmnist['train_labels']
	#valid_dataset = allmnist['valid_dataset']
	#valid_labels = allmnist['valid_labels']
	test_dataset = allmnist['test_dataset']
	test_labels = allmnist['test_labels']
	del allmnist  # free up memory

#print('Training set', train_dataset.shape, train_labels.shape)
#rint('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
#print('Test set', type(test_dataset), type(test_labels.shape))  ## test de los test

#test_dataset = test_dataset[:1,:,:]
#test_labels = test_labels[:1,]
#print('Test set', test_dataset.shape, test_labels.shape)

# Read model parameters ##
param_file = os.path.join(model_path, 'modeling/modelC3_param.pickle')
with open(param_file, 'rb') as f:
	param = pickle.load(f)
	#train_dataset = allmnist['train_dataset']
	#train_labels = allmnist['train_labels']
	#valid_dataset = allmnist['valid_dataset']
	#valid_labels = allmnist['valid_labels']
	weights = param['weights']
	biases = param['biases']
	del param  # free up memory

print('Weights', weights['w1'].shape)


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
def reformat(dataset, labels=None):
	dataset = dataset.reshape((-1, image_size, image_size, n_channels)).astype(np.float32)
	labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
	return dataset, labels

#train_dataset, train_labels = reformat(train_dataset, train_labels)
#valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
#print('Training set_', train_dataset.shape, train_labels.shape, type(train_dataset))
#print('Validation set_', valid_dataset.shape, valid_labels.shape, type(valid_dataset))
print('Test set_', test_dataset.shape, test_labels.shape, type(test_dataset))



# Visualize a random image (from original data)
ran_image = ran.randint(0, test_dataset.shape[0] - 1)
#print('ran image', ran_image, test_dataset.shape[0])  ## test para solo 1 caso
print('Random label: ' + str(test_labels[ran_image]))
print('Random image: ')
#print(train_dataset[ran_image])
label = test_labels[ran_image].argmax(axis=0)
image = test_dataset[ran_image].reshape([32,32])
plt.title('Example: %d Label: %d' % (ran_image, label))
#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.imshow(image, cmap=plt.cm.gray)
print(test_labels[ran_image].argmax(axis=0), '->', chr(test_labels[ran_image].argmax(axis=0)+ord('A')))
plt.show()




# Step 2: Define data constant ####
# each image in the allMNIST data is of shape 32*32 = 1024
# therefore, each image is represented with a 1x1024 tensor
# there are 26 classes for each image, corresponding to English letters A - Z. 
X_test = tf.constant(test_dataset)


# Step 3: create weights and bias ####
# weights and biases are random initialized
# shape of w depends on the dimension of X and the next layer (i.e so that Y = X * w + b)
# shape of b depends on the next layers (i.e Y)
#weights = {
#	'w1' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, n_channels, depth], stddev=0.1)),
#	'w2' : tf.Variable(tf.random_normal(shape=[patch_size, patch_size, depth, depth], stddev=0.1)),
#	'w3' : tf.Variable(tf.random_normal(shape=[image_size // 4 * image_size // 4 * depth, n_hidden], stddev=0.1)),
#	'w4' : tf.Variable(tf.random_normal(shape=[n_hidden, n_classes], stddev=0.1))
#}

#biases = {
#	'b1' : tf.Variable(tf.zeros([depth])),
#	'b2' : tf.Variable(tf.constant(1.0, shape=[depth])),
#	'b3' : tf.Variable(tf.constant(1.0, shape=[n_hidden])),
#	'b4' : tf.Variable(tf.constant(1.0, shape=[n_classes]))
#}


# Step 4: build model ####
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image

def convnet_model(X, weights, biases, keep_prob):  # 96.8% ModelC3
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

#logits = convnet_model(X)

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch
#entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
#loss = tf.reduce_mean(entropy)


# Step 6: define training op
# using gradient descent to minimize loss
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


# Accuracy function
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


# Step 7: Prediction*
#train_pred = tf.nn.softmax(logits)
#valid_pred = tf.nn.softmax(convnet_model(X_val))
test_pred = tf.nn.softmax(convnet_model(X_test, weights, biases, keep_prob=1.0))

with tf.Session() as sess:
	# to visualize using TensorBoard
	#writer = tf.summary.FileWriter('./graphs/convnet', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	#n_batches = int(train_dataset.shape[0]/batch_size)

	print('Testing initialized!')
	#for i in range(n_epochs): # train the model n_epochs times
	#	total_loss = 0
	preds_ = test_pred.eval()
	pred_label = tf.argmax(preds_, 1)
	pred_label_ = sess.run(pred_label)
	#	for j in range(n_batches):
	#		offset = (j * batch_size) % (train_labels.shape[0] - batch_size)
	#		X_batch = train_dataset[offset:(offset + batch_size),:,:,:]
	#		Y_batch = train_labels[offset:(offset + batch_size),:]
	#		# TO-DO: run optimizer + fetch loss_batch
	#		_, loss_batch, preds = sess.run([optimizer, loss, train_pred], feed_dict={X:X_batch, Y:Y_batch})

	#		total_loss += loss_batch
	#		if(j % 1000 == 0):
	#			print('		Minibatch loss at step %d: %f' % (j, loss_batch))
	#			print('		Minibatch accuracy: %.1f%%' % accuracy(preds, Y_batch))
	#			print('		Validation accuracy: %.1f%%' % accuracy(valid_pred.eval(), valid_labels))
	#	print('	Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_labels))
	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Testing Finished!')
	
	print('Labels, preds', test_labels[1:10,:], pred_label_[1:10])
	

	# Visualization of the predicctions on test data
	for _ in range(20):
		ran_image = ran.randint(0, test_dataset.shape[0] - 1)  # -1
		print('Random test label {0}'.format(test_labels[ran_image]))
		print(test_labels[ran_image].argmax(axis=0), '->', chr(test_labels[ran_image].argmax(axis=0)+ord('A')))
		

		preds_ = test_pred.eval()
		pred_label = tf.argmax(preds_, 1)
		print('Predicted value = %d' % sess.run(pred_label[ran_image]))

		label = test_labels[ran_image].argmax(axis=0)
		#image = test_dataset[ran_image].reshape([32,32])
		#plt.title('Example test: %d Label: %d' % (ran_image, label))
		#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
		#plt.show()

		image = test_dataset[ran_image].reshape([32,32])
		plt.title('Example test: %d Label: %d' % (ran_image, label))
		plt.imshow(image, cmap=plt.cm.gray)
		plt.show()