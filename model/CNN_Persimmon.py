import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # Since my defaulted keras backend is theano, I change to tensorflow here.
import tensorflow
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

datapath='/mnt/e/testjupyter/Permision_Detec' #Modify by yourself
savepath='/mnt/e/testjupyter/Permision_Detec/check' #Modify by yourself
X_train = np.load('{}/Persimmon.npy'.format(datapath)).astype('float32')/255.0 
y_train = np.load('{}/label.npy'.format(datapath))
y_trian_onehot = np_utils.to_categorical(y_train)

Xdims = X_train.shape[1:]

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(10,20), input_shape=Xdims, activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(10,20), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()

checkpoint = ModelCheckpoint('%s/weights-{epoch:02d}-{val_acc:.2f}.hdf5'%savepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=X_train, y=y_trian_onehot, validation_split=0.25,epochs=100, batch_size=128, callbacks = callbacks_list,verbose=1)
model.save('{}/my_model.h5'.format(savepath))
