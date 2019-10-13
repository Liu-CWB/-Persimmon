import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['THEANO_FLAGS'] = 'optimizer=None'
import tensorflow
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  
from keras.utils import np_utils 
from keras.callbacks import ModelCheckpoint
#import os
#os.environ['THEANO_FLAGS'] = 'optimizer=None'
savepath='/mnt/e/testjupyter/Permision_Detec/check'
X_train = np.load('/mnt/e/testjupyter/Permision_Detec/Persimmon.npy').astype('float32')/255.0
y_train = np.load('/mnt/e/testjupyter/Permision_Detec/label.npy')
y_trian_onehot = np_utils.to_categorical(y_train)
X_train = X_train[:,3:38,13:48,:]
Xdims = X_train.shape[1:]
print(Xdims)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=Xdims, activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
#model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
#model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
#model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
#model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
#model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
#model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
#model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()

checkpoint = ModelCheckpoint('%s/weights-{epoch:02d}-{acc:.2f}.hdf5'%savepath, monitor='acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=X_train, y=y_trian_onehot,epochs=100, batch_size=128, callbacks = callbacks_list,verbose=1)   
model.save('{}/my_model.h5'.format(savepath)) 
