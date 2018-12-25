import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os, os.path
import time

def pos2vel(pos):
    vel = np.zeros_like(pos)
    for i in range(1, len(pos)):
        vel[i] = pos[i] - pos[i-1]
    return vel

def vel2pos(vel):
    pos = np.zeros_like(vel)
    for i in range(1, len(vel)):
        pos[i] = pos[i-1] + vel[i-1]
    return pos

def readAttPosData(num):
    fName = 'Data/poses/'
    fName = fName + ( '0' + str(num) if num < 10 else str(num)) + '.txt'
    data = pd.read_csv(fName, sep=" ", header=None)
    data = data.as_matrix()
    DCM = data[:, [0,1,2, 4,5,6, 8,9,10]]
    pos = data[:,[3, 7, 11]]
    return DCM, pos, pos2vel(pos)

def getLabels(seq):
    DCM, pos, vel = readAttPosData(seq)
    DCM = DCM[1:,:]
    pos = pos[1:,:]
    vel = vel[1:,:]
    return DCM, pos, vel

def getOF(seq):
    fName = 'Data/Flow/OF_Full' + str(seq) + '.npy'
    data = np.load(fName)
    return data

def getData(seq):
    of = getOF(seq)
    DCM, pos, vel = getLabels(seq)
    return of, vel, pos, DCM

def getImage(seq):
    imgList = np.load('Data/Images/seq' + str(seq) + '.npy')
    img1List = imgList[:-1,:,:,:]
    img2List = imgList[1:,:,:,:]
    return img1List, img2List

def getMergedData(seqList):
    ofList = None
    velList = None
    posList = None
    DCMList = None
    imgList1 = []
    imgList2 = []
    totalImgNumber = 0
    d1,d2 = 0,0
    for seq in seqList:
        print ('Reading seq ' + str(seq))
        of, vel, pos, DCM = getData(seq)
        img1, img2 = getImage(seq)
        totalImgNumber += img1.shape[0]
        ofList = of if ofList is None else np.concatenate([ofList, of], axis=0)
        velList = vel if velList is None else np.concatenate([velList, vel], axis=0)
        posList = pos if posList is None else np.concatenate([posList, pos], axis=0)
        DCMList = DCM if DCMList is None else np.concatenate([DCMList, DCM], axis=0)
        imgList1.append(img1)
        imgList2.append(img2)
        print ('Done reading seq ' + str(seq))

    print ('%d data points are being prepared...' %(totalImgNumber))

    imgTotal1 = np.ndarray(shape=(totalImgNumber,360,640,3), dtype='float16')
    imgTotal2 = np.ndarray(shape=(totalImgNumber,360,640,3), dtype='float16')

    end = 0
    for i in range(0,len(seqList)):
        img1 = imgList1[i]
        img2 = imgList2[i]
        imgNum = img1.shape[0]
        start = end
        end = start+imgNum
        print ('changing data type and concatenating for seq %d ...' %(seqList[i]))
        imgTotal1[start:end,:,:,:] = img1.astype('float16')/255.0-0.5
        imgTotal2[start:end,:,:,:] = img2.astype('float16')/255.0-0.5

    print ('Done data prep.')
    return ofList, velList, posList, DCMList, imgTotal1, imgTotal2

if __name__ == '__main__':
    start = time.time()
    print ('starting...')
    of, vel, pos, DCM, img1, img2 = getMergedData([0,1,2, 4,8])

    # print of.shape
    # print vel.shape
    # print DCM.shape
    # print img1.shape
    # print img2.shape

    end = time.time()
    #print 'time elapsed: %f' %(end - start)
