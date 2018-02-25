import numpy as np
import pandas as pd
import os, os.path
import cv2

def getImgName(index):
    if index <10:
        name = '00000' + str(index) + '.png'
    elif index <100:
        name = '0000' + str(index) + '.png'
    elif index < 1000:
        name = '000' + str(index) + '.png'
    elif index < 10000:
        name = '00' + str(index) + '.png'
    elif index < 100000:
        name = '0' + str(index) + '.png'
    else:
        name = str(index) + '.png'
    return name

def getImage(seq, side=0):
    imgDir = '../../DLData/KITTI/odom/dataset/sequences/'
    if side==0:
        imgDir = imgDir + ( '0' + str(seq) if seq < 10 else str(seq)) + '/image_2/'
    else:
        imgDir = imgDir + ( '0' + str(seq) if seq < 10 else str(seq)) + '/image_3/'
    numFile = len([name for name in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, name))])

    index = 0
    flowData = []

    imgList = []

    while (index < numFile):
        fName_curr = imgDir + getImgName(index)
        img_curr = cv2.imread(fName_curr)
        img_curr = cv2.resize(img_curr, (1152,320))
        imgList.append(img_curr)
        index += 1
        if index%100==0:
            print index

    imgList = np.array(imgList)

    np.save('Data/Images/seq'+str(seq), imgList)
    print 'done'



if __name__ == '__main__':
    #getImage(5)
    for i in range(0,11):
        getImage(i)
