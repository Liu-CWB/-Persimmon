import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def resizegif(st,ed,theday):
    a = 419
    over = 'finish'
    for numgif in range(st,ed+1):
        if numgif < 10:
            rnum = '000'+str(numgif)
        elif numgif >= 10 and numgif < 100:
            rnum = '00'+ str(numgif)
        elif numgif >= 100 and numgif < 1000:
            rnum = '0' + str(numgif)
        else:
            rnum = str(numgif)
        try: 
            im = Image.open("E:/testjupyter/Permision_Detec/day{}_{}/DSCF{}.JPG".format(theday,theday,rnum))
            (width, height) = (im.width //64, im.height // 59)
            im_resized = im.resize((width, height))
            if a < 10:
                filename = '000' + str(a)
            elif a >= 10 and a < 100:
                filename = '00'+ str(a)
            elif a >= 100 and a < 1000:
                filename = '0'+ str(a)
            else:
                filename = str(a)
            im_resized.save('E:/testjupyter/Permision_Detec/resize_day{}/{}_{}.jpg'.format(theday,theday,filename))
            a= a + 1
        except:
            print('No such number:{}'.format(rnum))
    return over

resizegif(2998,3170,'four')

