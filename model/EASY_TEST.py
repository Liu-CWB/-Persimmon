import numpy as np
import pandas as pd
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras.models import load_model

np.set_printoptions(precision=2)
path='/mnt/e/testjupyter/Permision_Detec' # Modify by yourself
model = load_model('{}/check/first_my_model.h5'.format(path))

y = []
y_class = []
theday = ['one','two','three','four']
for day in theday:
    for i in range(551,591):
        im = Image.open('{}/testdata/{}_0{}.jpg'.format(path,day,str(i)))
        im = np.array(im).reshape(1,48,66,3).astype('float32')/255.0
        #y.append(model.predict(im))
        y_class.append(model.predict_classes(im))

#Create test label
for i in range(1,5):
    if i == 1:
        q = np.repeat(i,40)
    else:
        q = np.concatenate((q,np.repeat(i,40)))
print(q,q.shape)

#y = np.array(y)
y_class = np.array(y_class,dtype=int)
y_class = y_class+1
cfdf = pd.crosstab(q,y_class.reshape(-1),rownames=['label'],colnames=['detection'])
print(cfdf)
