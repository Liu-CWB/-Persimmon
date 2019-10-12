import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

days = ['one','two','three','four']

def comb(theday):
    for numgif in range(1,501):
        if numgif % 100 ==0:
            print(numgif)
        if numgif < 10:
            rnum = '000'+str(numgif)
        elif numgif >= 10 and numgif < 100:
            rnum = '00'+ str(numgif)
        elif numgif >= 100 and numgif < 1000:
            rnum = '0' + str(numgif)
        else:
            rnum = str(numgif)
        im = Image.open("E:/testjupyter/Permision_Detec/resize_day{}/{}_{}.JPG".format(theday,theday,rnum))
        if numgif == 1:
            grid = np.array(im)
        else:
            grid = np.concatenate((grid,im))
    return grid
a= 0 
for daythe in days:
    q=comb(daythe)
    if a == 0 :
        comfig = q
    else:
        comfig = np.concatenate((comfig,q))
    a = a + 1
np.save('Persimmon.npy',comfig)

#Create label
for qq in range(4):
    label = np.repeat(int(qq),500)
    if qq == 0:
        fnlabel = label
    else:
        fnlabel = np.concatenate((fnlabel,label))
np.save('label.npy',fnlabel)
