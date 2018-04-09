import time
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
#matplotlib inline
plt.rcParams['figure.figsize'] = (10,6)

from sklearn import datasets
from sklearn import cross_validation

## define data - train, test, validation
## ground truth should be a matrix with ones and zeros

data = datasets.load_digits()
X, y = data['data'], pd.get_dummies(data['target'])
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=1234)

session =            tf.InteractiveSession()

# define placeholder variables
NUM_OF_FEATURES =    64 #CHANGE TO REAL NUM OF FEATURES
NUM_OF_HIDDEN =      64 #MIGHT CHANGE
NUM_OF_CLASS =       13

x_ =                 tf.placeholder(tf.float32,shape =[None,NUM_OF_FEATURES])
y_ =                 tf.placeholder(tf.float32,shape =[None,NUM_OF_CLASS])

# define the network computation
W_input =            tf.Variable(tf.truncated_normal([NUM_OF_FEATURES,NUM_OF_HIDDEN], stddev=0.1))
b_hidden =           tf.Variable(tf.constant(0.1,shape=[NUM_OF_HIDDEN]))

W_hidden =           tf.Variable(tf.truncated_normal([NUM_OF_HIDDEN,NUM_OF_CLASS], stddev=0.1))
b_output =           tf.Variable(tf.constant(0.1,shape=[NUM_OF_CLASS]))

hidden_activation =  tf.nn.sigmoid(tf.matmul(x_,W_input) + b_hidden)
yhat =               tf.nn.softmax(tf.matmul(hidden_activation,W_hidden)+ b_output)

# degine Loss function
mse_loss =           tf.reduce_mean(tf.square( yhat - y_ ))

# compute accuracy computation
correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y_, 1))
accuracy =           tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# set up training method
train_step =         tf.train.AdamOptimizer(0.1).minimize(mse_loss)
session.run(tf.initialize_all_variables())

# train the model
n_iters = 1500
for i  in range(n_iters+1):
    # run througth an iteration of the training procces
    train_step.run(feed_dict={x_: X_train, y_: Y_train })

    # compute accuracy and Loss
    if i % 100 == 0:
        current_loss = mse_loss.eval(feed_dict={x_: X_test, y_: Y_test})
        current_acc = accuracy.eval(feed_dict={x_: X_test, y_: Y_test})
        print('Train step: %d, Loss: %.3f, Accuracy: %.3f'  %(i,current_loss,current_acc))



