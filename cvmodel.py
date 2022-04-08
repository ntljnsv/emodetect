#importing libraries
import matplotlib.pyplot as plt

#from keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#image size & location
pictureSize = 48
folderPath = (r'.\images\\')


batchSize = 128

datagenTrain = ImageDataGenerator()
datagenVal = ImageDataGenerator()

trainSet = datagenTrain.flow_from_directory(folderPath+'train', target_size=(pictureSize, pictureSize),
                                            color_mode='grayscale', batch_size=batchSize, class_mode='categorical',
                                            shuffle=True)

testSet = datagenVal.flow_from_directory(folderPath+'validation', target_size=(pictureSize, pictureSize),
                                         color_mode='grayscale', batch_size=batchSize, class_mode='categorical',
                                         shuffle=True)
#model building


noOfClasses = 7

model = Sequential()
#block 1
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#block 2
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#block 3
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#block 4
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#block 5
model.add(Flatten())

#fully connected 1 layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

#block 6
#fully connected 2 layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#block 7
model.add(Dense(noOfClasses, activation='softmax'))

opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


checkpoint = ModelCheckpoint('cvmodel.py', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)

reduceLearningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)

callbacksList = [earlyStopping, checkpoint, reduceLearningRate]

epochs = 48

history = model.fit_generator(generator=trainSet, steps_per_epoch=trainSet.n//trainSet.batch_size, epochs=epochs,
                              validation_data=testSet, validation_steps=testSet.n//testSet.batch_size,
                              callbacks=callbacksList)
model.save('cvmodel.h5')

#plotting accuracy
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.ylabel('Accuracy ', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
