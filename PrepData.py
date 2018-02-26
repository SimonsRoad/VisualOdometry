import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os, os.path
import cv2

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
    imgList = imgList/255.0 - 0.5
    img1 = imgList[:-1,:,:,:]
    img2 = imgList[1:,:,:,:]
    return img1, img2

def getMergedData(seqList):
    ofList = None
    velList = None
    posList = None
    DCMList = None
    img1List = None
    img2List = None

    for seq in seqList:
        print 'Reading seq ' + str(seq)
        of, vel, pos, DCM = getData(seq)
        img1, img2 = getImage(seq)
        ofList = of if ofList is None else np.concatenate([ofList, of], axis=0)
        velList = vel if velList is None else np.concatenate([velList, vel], axis=0)
        posList = pos if posList is None else np.concatenate([posList, pos], axis=0)
        DCMList = DCM if DCMList is None else np.concatenate([DCMList, DCM], axis=0)
        img1List = img1 if img1List is None else np.concatenate([img1List, img1], axis=0)
        img2List = img2 if img2List is None else np.concatenate([img2List, img2], axis=0)
        print 'Done reading seq ' + str(seq)

    return ofList, velList, posList, DCMList, img1List, img2List

if __name__ == '__main__':
    of, vel, DCM, img1, img2 = getMergedData([2])
    print of.shape
    print vel.shape
    print DCM.shape
    print img1.shape
    print img2.shape


    #end
