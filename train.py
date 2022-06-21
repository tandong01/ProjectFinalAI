import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras import datasets,layers,models
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD,RMSprop,Adam
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.callbacks import ModelCheckpoint

dataset_path = 'data/'

image_size = (128,128)
TRAINING_DATA_DIR = str(dataset_path)
print(TRAINING_DATA_DIR)
# modify data
kwargs_datagen = dict(rescale=1./255, validation_split=0.2) # 20 percent for validation
# validation data modify 
valid_datagen = ImageDataGenerator(**kwargs_datagen)
valid_generator = valid_datagen.flow_from_directory(TRAINING_DATA_DIR, subset="validation", shuffle=True, target_size=image_size)
# train data modify
train_datagen = ImageDataGenerator(**kwargs_datagen)
train_generator = train_datagen.flow_from_directory(TRAINING_DATA_DIR,subset="training",shuffle=True,target_size=image_size)

image_batch_train, label_batch_train = next(iter(train_generator))
print("image batch shape: ", image_batch_train.shape)
print("label batch shape: ", label_batch_train.shape)
dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print("labels: ", dataset_labels)
print("match class: ", train_generator.class_indices)

# Create layer model with CNN

model1=Sequential()
# BLock 1
model1.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same',input_shape=(128,128,3))) 
model1.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same')) 
model1.add(MaxPooling2D((2,2)))
#Block 2
model1.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same')) 
model1.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same'))
model1.add(MaxPooling2D((2,2)))
#Block 3
model1.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same')) 
model1.add(Conv2D(128,(3,3),activation='relu',kernel_initializer='he_uniform',padding='same')) 
model1.add(MaxPooling2D((2,2)))

model1.add(Flatten())
model1.add(Dense(512,activation='relu',kernel_initializer='he_uniform'))
model1.add(Dense(128,activation='relu'))
model1.add(Dense(15,activation='softmax'))
model1.summary()


#Training
opt=SGD(learning_rate=0.01,momentum=0.9)
model1.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
history1=model1.fit(train_generator,epochs=50,batch_size=4,validation_data=valid_generator,steps_per_epoch=steps_per_epoch,verbose=1)

frame = "wieghts-test-CNN1.hdf5"
model1.save_weights(frame, overwrite = True )

# Save model
model1.save('ANIMAL4.h5')

# danh gia do chinh xac cua CNN
score = model1.evaluate(train_generator, verbose = 0)
print('sai so kiem tra la: ', score[0])
print('do chinh xac kiem tra la: ', score[1])

# Diagram
pd.DataFrame(history1.history).plot(figsize = (18,5))
plt.grid(True)
plt.gca().set_ylim(0,10)
plt.show()

plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc = 'upper left')
plt.show()

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc = 'upper left')
plt.show()