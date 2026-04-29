import os
import random
import warnings
warnings.filterwarnings("ignore")
from utils import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# ** ADD YOUR CODE HERE **
filter_size = (3,3) 
filter_num = 32
input_size = (180, 180)
pool_size = (2,2)

model.add(Conv2D(filter_num, filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Conv2D(filter_num, filter_size, activation='relu'))
model.add(MaxPooling2D(pool_size = pool_size))

model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="sigmoid"))


print("Model compiles")
print(model) 

