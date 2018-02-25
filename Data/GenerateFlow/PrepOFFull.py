import cv2
import numpy as np
import os, os.path
import time

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



def imshow(img, windowName='dummy'):
    # cv2.imshow('left_prev', img_prev)
    cv2.imshow(windowName, img)
    k = cv2.waitKey(1)
    return k

def GetOF(seq=0, h_reduc = 10, w_reduc = 10, side=0):
    imgDir = '../../DLData/KITTI/odom/dataset/sequences_grey/'
    if side==0:
        imgDir = imgDir + ( '0' + str(seq) if seq < 10 else str(seq)) + '/image_0/'
    else:
        imgDir = imgDir + ( '0' + str(seq) if seq < 10 else str(seq)) + '/image_1/'
    numFile = len([name for name in os.listdir(imgDir) if os.path.isfile(os.path.join(imgDir, name))])

    index = 1
    flowData = []

    while (index < numFile):
        fName_prev = imgDir + getImgName(index-1)
        fName_curr = imgDir + getImgName(index)
        img_prev = cv2.imread(fName_prev, 0)
        img_curr = cv2.imread(fName_curr, 0)

        img_prev = cv2.resize(img_prev, (36, 10))
        img_curr = cv2.resize(img_curr, (36, 10))

        flow = cv2.calcOpticalFlowFarneback(img_prev, img_curr, None, 0.5, 3, 10, 1, 1, 1.2, 0)

        flowData.append(flow)

        # if imshow(img_curr,'dd')==113: #'q'
        #     break
        if index%100 ==0 :
            print index
        index += 1

    flowData = np.array(flowData)
    print flowData.shape
    return flowData

def getOFReady(seq):

    start = time.time()
    folderName = 'Data/Flow/'

    OF116936_l = GetOF(seq, 2, 4, 0)
    np.save(folderName + '/OF_Full'+str(seq), OF116936_l)

    end = time.time()
    print(end - start)


if __name__ == '__main__':
    getOFReady(5)
