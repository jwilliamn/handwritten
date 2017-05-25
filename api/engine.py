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
import time

# Global settings ####
# Path to saved model parameters
# model_path = '/home/williamn/Repository/handwritten'
model_path = ''
# model_path = '/home/vmchura/Documents/handwritten'

# Parameters model
# learning_rate = 0.05
# batch_size = 16  # 100
# patch_size = 5
# n_epochs = 2 # 15

# Network Parameters
image_size = 32
# n_input = 1024  # input nodes
# depth = 16
n_channels = 1  # grayscale


# n_hidden = 64  # hidden layer nodes
# n_classes = 26  # total classes (A - Z)


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
    # labels = (np.arange(n_classes) == labels[:,None]).astype(np.float32)
    return dataset, labels


class Engine:
    def __init__(self, arg, pickle_path):
        # param_file = os.path.join(arg, 'modeling/modelD2_param.pickle')
        param_file = os.path.join(arg, pickle_path)
        weights, biases = getModelParams(param_file)
        print('opens it')
        self.weights = weights
        self.biases = biases
        self.data = []
        self.pred = []

    def push(self, imageData):
        self.data.append(imageData)
        return len(self.data) - 1

    def runEngine(self):
        print('N Data: ', len(self.data))
        if len(self.data) == 0:
            print('Predichos: ', len(self.pred))
            return
        weights = self.weights
        biases = self.biases
        valid_dataset, _ = make_arrays(len(self.data), image_size)
        for k, valid_data in enumerate(self.data):
            valid_dataset[k:(k + image_size), ::] = valid_data
        image_dataset, _ = reformat(valid_dataset, labels=None)

        X = tf.constant(image_dataset)

        image_pred = tf.nn.softmax(convnet_model(X, weights, biases, keep_prob=1.0))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            preds_ = image_pred.eval()
            pred_label = tf.argmax(preds_, 1)
            pred_label_ = sess.run(pred_label)

            self.pred = pred_label_
        print('Predichos: ', len(self.pred))


class UniqueEngineDigit(Engine):
    def __new__(cls, arg=None):
        if not UniqueEngineDigit.instance:
            UniqueEngineDigit.instance = Engine(arg[0], arg[1])
        return UniqueEngineDigit.instance
    instance = None


class UniqueEngineLetter(Engine):
    def __new__(cls, arg=None):
        if not UniqueEngineLetter.instance:
            UniqueEngineLetter.instance = Engine(arg[0], arg[1])
        return UniqueEngineLetter.instance
    instance = None


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


# Trained models ####
# Characters
def convnet_model_old(X, weights, biases):  # 95.4% ModelC1
    # 1st Convolution layer
    conv = tf.nn.conv2d(X, weights['w1'], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + biases['b1'])
    hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 2nd Convolution layer
    conv = tf.nn.conv2d(hidden, weights['w2'], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + biases['b2'])
    hidden = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights['w3']) + biases['b3'])

    # Output
    o_layer = tf.matmul(hidden, weights['w4']) + biases['b4']
    return o_layer


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
    h_conv3_flat = tf.reshape(h_conv3, [shape[0], shape[1] * shape[2] * shape[3]])
    h_fllc1 = tf.nn.relu(tf.matmul(h_conv3_flat, weights['w4']) + biases['b4'])

    # Dropout
    h_fllc1_drop = tf.nn.dropout(h_fllc1, keep_prob)

    # ReadOut layer
    o_layer = tf.matmul(h_fllc1_drop, weights['w5']) + biases['b5']
    return o_layer


# Digits
def convnet_model_d_old(X, weights, biases, keep_prob):  # ModelD1 97.4%
    # 1st Convolution layer
    conv1 = tf.nn.conv2d(X, weights['w1'], [1, 1, 1, 1], padding='SAME')
    h_conv1 = tf.nn.relu(conv1 + biases['b1'])
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 2nd Convolution layer
    conv2 = tf.nn.conv2d(h_pool1, weights['w2'], [1, 1, 1, 1], padding='SAME')
    h_conv2 = tf.nn.relu(conv2 + biases['b2'])
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    shape = h_pool2.get_shape().as_list()
    h_pool2_flat = tf.reshape(h_pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    h_fllc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights['w3']) + biases['b3'])

    # Dropout
    h_fllc1_drop = tf.nn.dropout(h_fllc1, keep_prob)

    # ReadOut layer
    o_layer = tf.matmul(h_fllc1_drop, weights['w4']) + biases['b4']
    return o_layer


def convnet_model_d(X, weights, biases, keep_prob):  # 98.2% modelD2
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
    h_conv3_flat = tf.reshape(h_conv3, [shape[0], shape[1] * shape[2] * shape[3]])
    h_fllc1 = tf.nn.relu(tf.matmul(h_conv3_flat, weights['w4']) + biases['b4'])

    # Dropout
    h_fllc1_drop = tf.nn.dropout(h_fllc1, keep_prob)

    # ReadOut layer
    o_layer = tf.matmul(h_fllc1_drop, weights['w5']) + biases['b5']
    return o_layer


# Prediction of characters ####
def predictImage(image_data):
    #print('Real image', image_data.shape, type(image_data))

    # Read model parameters ##
    # param_file = os.path.join(model_path, 'modeling/modelC1_param.pickle')
    # start_time = time.time()
    LetterPredictor = UniqueEngineLetter([model_path, 'modeling/modelC3_param.pickle'])
    return LetterPredictor.push(image_data)


#
# weights = LetterPredictor.weights
# biases = LetterPredictor.biases
# print("loading Weigths and Biases %s seconds ---" % (time.time() - start_time))
# # Reformat image to convnet shape
# start_time = time.time()
# image_dataset, _ = reformat(image_data, labels=None)
# print("reformating %s seconds ---" % (time.time() - start_time))
# # Define data constant ####
# start_time = time.time()
# X = tf.constant(image_dataset)
# print("tf.constant %s seconds ---" % (time.time() - start_time))
# # Pred run
# #image_pred = tf.nn.softmax(convnet_model(X, weights, biases))
# start_time = time.time()
# image_pred = tf.nn.softmax(convnet_model(X, weights, biases, keep_prob=1.0))
# print("tf.nn.softmax %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# with tf.Session() as sess:
# 	start_time = time.time()
# 	sess.run(tf.global_variables_initializer())
#
# 	#print('Prediction initialized!')
#
# 	preds_ = image_pred.eval()
# 	pred_label = tf.argmax(preds_, 1)
# 	pred_label_ = sess.run(pred_label)
# 	#print('Predicted value = %d' % sess.run(pred_label))
# 	#print(sess.run(pred_label), '->', chr(sess.run(pred_label)+ord('A')))
#
#
# 	#print('Total time: {0} seconds'.format(time.time() - start_time))
# 	#print('Testing Finished!')
#
# 	#image = image_data.reshape([32,32])
# 	#plt.title('Image - FSU')
# 	#plt.imshow(image, cmap=plt.cm.gray)
# 	#plt.show()
# print("with tf.session %s seconds ---" % (time.time() - start_time))
# return pred_label_

def initEngines():
    LetterPredictor = UniqueEngineLetter([model_path, 'modeling/modelC3_param.pickle'])
    DigitPredictor = UniqueEngineDigit([model_path, 'modeling/modelD2_param.pickle'])
# Prediction of digits ####
def predictImageDigit(image_data):
    #print('Real digit image', image_data.shape, type(image_data))

    # Read model parameters ##
    # param_file = os.path.join(model_path, 'modeling/modelD1_param_.pickle')
    # param_file = os.path.join(model_path, 'modeling/modelD1_param.pickle')
    DigitPredictor = UniqueEngineDigit([model_path, 'modeling/modelD2_param.pickle'])
    return DigitPredictor.push(image_data)

# weights  = DigitPredictor.weights
# biases = DigitPredictor.biases
#
# # Reformat image to convnet shape
# image_dataset, _ = reformat(image_data, labels=None)
#
# # Define data constant ####
# X = tf.constant(image_dataset)
#
# # Pred run
# image_pred = tf.nn.softmax(convnet_model_d(X, weights, biases, keep_prob=1.0))
#
# with tf.Session() as sess:
# 	start_time = time.time()
# 	sess.run(tf.global_variables_initializer())
#
# 	#print('Prediction initialized!')
#
# 	preds_ = image_pred.eval()
# 	pred_label = tf.argmax(preds_, 1)
# 	pred_label_ = sess.run(pred_label)
# 	#print('Predicted value = %d' % sess.run(pred_label))
# 	#print(sess.run(pred_label), '->', chr(sess.run(pred_label)+ord('A')))
#
#
# 	#print('Total time: {0} seconds'.format(time.time() - start_time))
# 	#print('Testing Finished!')
#
# 	#image = image_data.reshape([32,32])
# 	#plt.title('Image - FSU')
# 	#plt.imshow(image, cmap=plt.cm.gray)
# 	#plt.show()
#
# return pred_label_
#
