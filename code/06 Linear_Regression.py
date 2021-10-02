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
x_train = tf.random.uniform([100], minval=-1, maxval=1).numpy()
y_train = x_train
x_test = tf.random.uniform([100], minval=-1, maxval=1).numpy()
y_test = x_test

#get one random batch of data
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs], y_data[idxs]
    
# Python optimisation variables
epochs = 10
batch_size = 10

# convert x_test to tensor to pass through model (train data will be converted to
# tensors on the fly)
x_test = tf.Variable(x_test)

# training variables
m = tf.Variable(tf.random.uniform([], minval=-1, maxval=1))
n = tf.Variable(tf.random.uniform([], minval=-1, maxval=1))
# ~ m = tf.Variable([-1.0])
# ~ n = tf.Variable([1.0])

#feed forward function
def nn_model(x_input, m, n):
  logits = x_input * m + n
  return logits

#loss function
def loss_fn(logits, labels):
  mse = tf.keras.losses.MeanSquaredError()
  loss = mse(logits, labels)
  return loss

# setup the optimizer
optimizer = tf.keras.optimizers.SGD()

# == main training loop ==

total_batch = int(len(y_train) / batch_size)

#for epoch in range(epochs): # commented out to make way for animation
fig = plt.figure()
def animate(self):
  avg_loss = 0
  for i in range(total_batch):
    #get batch
    batch_x, batch_y = get_batch(x_train, y_train, batch_size)
    # create tensors
    batch_x = tf.Variable(batch_x)
    batch_y = tf.Variable(batch_y)
    #find gradients
    with tf.GradientTape() as tape:
      logits = nn_model(batch_x, m, n)
      loss = loss_fn(logits, batch_y)
    gradients = tape.gradient(loss, [m, n])
    #optimise gradients
    optimizer.apply_gradients(zip(gradients, [m, n]))
    # calculate loss
    avg_loss += loss / total_batch
  #test the test data
  test_logits = nn_model(x_test, m, n)
  test_loss = loss_fn(test_logits, y_test)
  print(f"Epoch: {0}, loss={avg_loss:.3f}, test set loss={test_loss:.3f}")
  #VISUALIZE
  print("n = ", n.numpy(), "m = ", m.numpy())
  fig.add_artist(lines.Line2D([-1, 1], [nn_model(-1, m, n), nn_model(1, m, n)]))  
  
ani = FuncAnimation(fig, animate, frames=1000, interval=10, repeat=False)
plt.show()





