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
model.add(Dense(1, activation="sigmoid"))


print("Model compiles")
print(model) 

model.compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = ['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory(
  'data/Train',
  target_size = input_size,
  batch_size = 32,
  class_mode = 'binary'
)

test_set = test_datagen.flow_from_directory(
  'data/Test',
  target_size = input_size,
  batch_size = 32,
  class_mode = 'binary'
)

model.fit(
  train_set,
  steps_per_epoch = 100,
  epochs = 10
)

loss, accuracy = model.evaluate(test_set, steps = 50)
print("Test Accuracy", accuracy)
