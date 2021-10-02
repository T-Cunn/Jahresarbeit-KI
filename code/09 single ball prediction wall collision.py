#!/usr/bin/env python3


#import tensorflow and stuff
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as lines
import numpy as np

#create training data
x_train = np.loadtxt("X.csv", delimiter=",")
y_train = np.loadtxt("y.csv", delimiter=",")


#get one random batch of data
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs], y_data[idxs]
    
# Python optimisation variables
epochs = 100000
batch_size = 1000

# training variables
w1 = tf.Variable(tf.random.uniform([4, 10], minval=-1, maxval=1), name="w1")
b1 = tf.Variable(tf.random.uniform([10], minval=-1, maxval=1), name="b1")
w2 = tf.Variable(tf.random.uniform([10, 4], minval=-1, maxval=1), name="w2")
b2 = tf.Variable(tf.random.uniform([4], minval=-1, maxval=1), name="b2")
# ~ w1 = tf.Variable([[1.0, -1.0], [0.0, 1.0]], name="w1")
# ~ b1 = tf.Variable([0.0, 0.0], name="w1")
# ~ w2 = tf.Variable([[0.0], [0.0]], name="w1")
# ~ b2 = tf.Variable([0.0], name="w1")


#feed forward function
def nn_model(data, w1, b1, w2, b2):
  data = tf.cast(data, tf.float32)
  data = tf.keras.activations.swish(tf.add(tf.matmul(data, w1), b1))
  data = tf.keras.activations.linear(tf.add(tf.matmul(data, w2), b2))
  return data

#loss function
def loss_fn(logits, labels): 
  bce = tf.keras.losses.MeanSquaredError()
  cross_entropy = bce(labels, logits)
  return cross_entropy

# setup the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

# == main training loop ==

total_batch = int(len(y_train) / batch_size)

epoch = 0
for epoch in range(epochs): # commented out to make way for animation
#def animate(self):
 #global epoch
  avg_loss = 0
  for i in range(total_batch):
    #get batch
    #batch_x, batch_y = get_batch(x_train, y_train, batch_size)
    # create tensors
    batch_x = tf.Variable(x_train)
    batch_y = tf.Variable(y_train)
    #find gradients
    with tf.GradientTape() as tape:
      logits = nn_model(batch_x, w1, b1, w2, b2)
      loss = loss_fn(logits, batch_y)
    gradients = tape.gradient(loss, [w1, b1, w2, b2])
    #optimise gradients
    optimizer.apply_gradients(zip(gradients, [w1, b1, w2, b2]))
    # calculate loss
    avg_loss += loss / total_batch
  #test the test data
  #test_logits = nn_model(x_test, w, b)
  #test_loss = loss_fn(test_logits, y_test)
  print(f"Epoch: {epoch}, loss={avg_loss:.3f}")

testVal = x_train[0]
print(testVal, " -> ", nn_model([[testVal]], w1, b1, w2, b2).numpy())
