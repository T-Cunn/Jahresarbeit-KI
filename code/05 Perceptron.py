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
m_target = 1
n_target = 0

x_train = tf.random.uniform([1000, 2], minval=-1, maxval=1).numpy()
y_train = x_train[:, 0] > x_train[:, 1]*m_target+n_target
y_train = tf.math.sign(y_train - 0.5).numpy()

x_test = tf.random.uniform([500, 2], minval=-1, maxval=1).numpy()
y_test = x_test[:, 0] > x_test[:, 1]*m_target+n_target
y_test = tf.math.sign(y_test - 0.5).numpy()

#setup drawing area
fig = plt.figure(figsize=(8, 8))

#draw data
for i in range(len(x_test)):
  if(y_test[i] > 0):
    fig.add_artist(plt.Circle((x_test[i, 0]/2+0.5, x_test[i, 1]/2+0.5), 0.02, color="red"))
  else:
    fig.add_artist(plt.Circle((x_test[i, 0]/2+0.5, x_test[i, 1]/2+0.5), 0.02, color="blue"))

#get one random batch of data
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs], y_data[idxs]
    
# Python optimisation variables
epochs = 10
batch_size = 100

# training variables
w = tf.Variable(tf.random.uniform([2], minval=-1, maxval=1))
b = tf.Variable(tf.random.uniform([], minval=-1, maxval=1))
# ~ m = tf.Variable([-1.0])
# ~ n = tf.Variable([1.0])


#feed forward function
def nn_model(x_input, w, b):
  logits = x_input * w
  logits = tf.math.reduce_sum(logits, axis = 1) + b
  output = tf.nn.sigmoid(logits)*2-1
  return output

#loss function
def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.keras.losses.mae(labels, logits))
    return cross_entropy

# setup the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# == main training loop ==

total_batch = int(len(y_train) / batch_size)

#for epoch in range(epochs): # commented out to make way for animation
epoch = 0
def animate(self):
  global epoch
  avg_loss = 0
  for i in range(total_batch):
    #get batch
    batch_x, batch_y = get_batch(x_train, y_train, batch_size)
    # create tensors
    batch_x = tf.Variable(batch_x)
    batch_y = tf.Variable(batch_y)
    #find gradients
    with tf.GradientTape() as tape:
      logits = nn_model(batch_x, w, b)
      loss = loss_fn(logits, batch_y)
    gradients = tape.gradient(loss, [w, b])
    #optimise gradients
    optimizer.apply_gradients(zip(gradients, [w, b]))
    # calculate loss
    avg_loss += loss / total_batch
  #test the test data
  #test_logits = nn_model(x_test, w, b)
  #test_loss = loss_fn(test_logits, y_test)
  print(f"Epoch: {epoch}, loss={avg_loss:.3f}")
  #VISUALIZE test
  if epoch % 10 == 0:
    results = nn_model(x_test, w, b)
    for i in range(len(results)):
      if(results[i] > 0):
        fig.add_artist(plt.Circle((x_test[i, 0]/2+0.5, x_test[i, 1]/2+0.5), 0.01, color="red"))
      else:
        fig.add_artist(plt.Circle((x_test[i, 0]/2+0.5, x_test[i, 1]/2+0.5), 0.01, color="blue"))
  epoch += 1
  
ani = FuncAnimation(fig, animate, frames=1000, interval=0, repeat=False)
plt.show()
