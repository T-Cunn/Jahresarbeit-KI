#!/usr/bin/env python3

#import tensorflow
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation

#import data
data = []
datadir = "/home/linuxuser/Projects_Current/12.klass Arbeit/code/drawing_shapes/"
categorys = ["hexahedron", "tetrahedron", "octahedron"]
catcount = len(categorys)
i = 0
sequence_length = 10
for cat in categorys:
  path = os.path.join(datadir, cat)
  temp = []
  imglist = os.listdir(path)
  imglist.sort()
  for img in imglist:
    img_array = np.amax(mpimg.imread(os.path.join(path, img)), 2)
    temp.append(np.ndarray.tolist(img_array))
    if len(temp) >= sequence_length:
      data.append([temp[:], i])
      del temp[0]
  i+=1
random.shuffle(data)
print(len(data))
x_train = []
y_train = []
for X, Y in data[1000:]:
  x_train.append(X)
  y_train.append(Y)
x_test = []
y_test = []
for X, Y in data[:1000]:
  x_test.append(X)
  y_test.append(Y)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
  
# convert x_test to tensor to pass through model (train data will be converted to tensors on the fly)
x_test = tf.Variable(x_test)
    
# Python optimisation variables
epochs = 50
batch_size = 100

# Weights
W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random.normal([300]), name='b1')
W2 = tf.Variable(tf.random.normal([300, catcount], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random.normal([catcount]), name='b2')


#feed forward function
def nn_model(x_input, W1, b1, W2, b2, verbose = False):
  # RNN state
  lastout = tf.zeros([catcount])
  # select images from the list for rnn
  for img in tf.unstack(x_input, axis=1):
    # flatten
    img = tf.reshape(img, (x_input.shape[0], -1))
    # 1st layer
    x = tf.add(tf.matmul(tf.cast(img, tf.float32), W1), b1)
    x = tf.nn.relu(x)
    # last layer + modifyer from last output
    logits = tf.add(tf.matmul(x, W2), b2)
    logits = tf.add(logits, tf.divide(lastout, 1))
    lastout = logits
    if verbose :
      print(np.round(tf.nn.softmax(logits).numpy(), 2)*100)
  return logits

#loss function
def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return cross_entropy

# setup the optimizer
optimizer = tf.keras.optimizers.Adam()

#get one random batch of data
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    return x_data[idxs,:,:], y_data[idxs]

#main training loop
total_batch = int(len(y_train) / batch_size)
for epoch in range(epochs):
    avg_loss = 0
    for i in range(total_batch):
        batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
        # create tensors
        batch_x = tf.Variable(batch_x)
        batch_y = tf.Variable(batch_y)
        # create a one hot vector
        batch_y = tf.one_hot(batch_y, catcount)
        with tf.GradientTape() as tape:
            logits = nn_model(batch_x, W1, b1, W2, b2)
            loss = loss_fn(logits, batch_y)
        gradients = tape.gradient(loss, [W1, b1, W2, b2])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
        avg_loss += loss / total_batch
    test_logits = nn_model(x_test, W1, b1, W2, b2)
    max_idxs = tf.argmax(test_logits, axis=1)
    test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
    print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set      accuracy={test_acc*100:.3f}%")
print("\nTraining complete!")

shapeNames = ["cube", "tetra", "octo"]

while(True):
  test = np.random.randint(0, len(y_test))
  output = tf.nn.softmax(nn_model(tf.expand_dims(x_test[test], 0), W1, b1, W2, b2, True))
  print("---")
  print("should be:", shapeNames[y_test[test]])
  print("AI thinks:", shapeNames[tf.argmax(output, 1).numpy()[0]])
  print("AI is", str(round(tf.reduce_max(output).numpy()*100, 2)) + "% sure")
  fig = plt.figure()
  sample = []
  for img in x_test[test]:
    sample.append([plt.imshow(img.numpy())])
  ani = animation.ArtistAnimation(fig, sample, interval=100, repeat=True)
  plt.show()
