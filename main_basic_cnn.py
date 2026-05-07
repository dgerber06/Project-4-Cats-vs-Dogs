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
input_size = (100, 100)

model.add(Conv2D(32, (3,3), activation='relu', input_shape = (100, 100, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = ['accuracy']
)

print("Model compiles")
print(model) 

train_datagen = ImageDataGenerator(rescale = 1./255)
train_set = train_datagen.flow_from_directory(
  'data/Train',
  target_size = input_size,
  batch_size = 64,
  class_mode = 'binary'
)

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
  'data/Test',
  target_size = input_size,
  batch_size = 64,
  class_mode = 'binary'
)

model.fit(
  train_set,
  steps_per_epoch = 200,
  epochs = 10,
  validation_data = test_set,
  validation_steps = 75
)

loss, accuracy = model.evaluate(test_set, steps = 75)
print("Test Accuracy", accuracy)
