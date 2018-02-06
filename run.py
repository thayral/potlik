import os
import numpy as np
import tensorflow as tf
import scipy.io
from matplotlib import pyplot as plt


# this is going to be some CNN stuff
#svhn digit cinvenet
#reverse engineering depuis tensorpack


print("nothing implemented yet...") # in python3




# dataset should be in /data
#SVHN_URL = "http://ufldl.stanford.edu/housenumbers/"


current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir,"data")

# Train dataset load
filename = os.path.join(data_dir, "train" + '_32x32.mat')
data = scipy.io.loadmat(filename)
train_X = data['X'].transpose(3, 0, 1, 2)
train_Y = data['y'].reshape((-1))
train_Y[train_Y == 10] = 0

#test dataset load
filename = os.path.join(data_dir, "test" + '_32x32.mat')
data = scipy.io.loadmat(filename)
test_X = data['X'].transpose(3, 0, 1, 2)
test_Y = data['y'].reshape((-1))
test_Y[test_Y == 10] = 0

n= train_X.shape[0]


#shape X : index, size1, size2, rgb

ind=45

print(train_Y[ind])

plt.imshow(train_X[ind], interpolation='nearest')
plt.show()
