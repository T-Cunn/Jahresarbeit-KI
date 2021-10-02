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
x_train = tf.random.uniform([1000, 2], minval=-1, maxval=1).numpy()
y_train = (x_train[:, 0] > x_train[:, 1]-0.5) & (x_train[:, 0] < x_train[:, 1]+0.5)

#setup drawing area
fig = plt.figure(figsize=(6, 6))

#draw data
for i in range(1000):
  if(y_train[i] > 0.5):
    fig.add_artist(plt.Circle((x_train[i, 0]/2+0.5, x_train[i, 1]/2+0.5), 0.02, color="red"))
  else:
    fig.add_artist(plt.Circle((x_train[i, 0]/2+0.5, x_train[i, 1]/2+0.5), 0.02, color="blue"))

#get one random batch of data
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs], y_data[idxs]
    
# Python optimisation variables
epochs = 10000
batch_size = 1000

# training variables
w1 = tf.Variable(tf.random.uniform([2, 2], minval=-1, maxval=1), name="w1")
b1 = tf.Variable(tf.random.uniform([2], minval=-1, maxval=1), name="b1")
w2 = tf.Variable(tf.random.uniform([2, 1], minval=-1, maxval=1), name="w2")
b2 = tf.Variable(tf.random.uniform([], minval=-1, maxval=1), name="b2")
# ~ w1 = tf.Variable([[1.0, -1.0], [0.0, 1.0]], name="w1")
# ~ b1 = tf.Variable([0.0, 0.0], name="w1")
# ~ w2 = tf.Variable([[0.0], [0.0]], name="w1")
# ~ b2 = tf.Variable([0.0], name="w1")


#feed forward function
def nn_model(x_input, w1, b1, w2, b2):
  layer1 = tf.nn.relu(tf.add(tf.matmul(tf.cast(x_input, tf.float32), w1), b1))
  output = tf.nn.sigmoid(tf.add(tf.matmul(layer1, w2), b2))
  return output

#loss function
def loss_fn(logits, labels): 
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
  cross_entropy = bce(labels, logits)
  return cross_entropy

# setup the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

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
  
#ani = FuncAnimation(fig, animate, frames=1000, interval=0, repeat=False)
#VISUALIZE test
results = nn_model(x_train, w1, b1, w2, b2)
for i in range(1000):
  a = round(results[i, 0].numpy())
  b = tf.cast(y_train[i], tf.int32).numpy()
  if(a > 0.5):
    fig.add_artist(plt.Circle((x_train[i, 0]/2+0.5, x_train[i, 1]/2+0.5), 0.01, color="red"))
  else:
    fig.add_artist(plt.Circle((x_train[i, 0]/2+0.5, x_train[i, 1]/2+0.5), 0.01, color="blue"))
  # ~ if(a == b):
    # ~ fig.add_artist(plt.Circle((x_train[i, 0]/2+0.5, x_train[i, 1]/2+0.5), 0.01, color="green"))
  # ~ else:
    # ~ fig.add_artist(plt.Circle((x_train[i, 0]/2+0.5, x_train[i, 1]/2+0.5), 0.01, color="black"))
plt.show()
