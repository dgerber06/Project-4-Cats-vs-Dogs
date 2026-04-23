import os
import random
import warnings
warnings.filterwarnings("ignore")
from utils import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# ** ADD YOUR CODE HERE **

filter_size = 9
filter_num = 64
input_size = (28,28,1)
pool_size = (2,2) 


model.add(Conv2D(filter_num, filter_size, input_shape=input_size, activation='relu', name="layer1")
model.add(MaxPooling2D(pool_size=pool_size)
model.add(Conv2D(filter_num, filter_size, input_shape=input_size, activation='relu', name="layer2")
model.add(MaxPooling2D(pool_size=pool_size)
model.add(Flatten())
model.add(Dense(4, name="full_connected_layer1"))
model.add(Dense(2, name="full_connceted_layer2"))



