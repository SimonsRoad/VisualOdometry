import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def readGT(num):
    fName = 'Data/poses/'
    fName = fName + ( '0' + str(num) if num < 10 else str(num)) + '.txt'
    data = pd.read_csv(fName, sep=" ", header=None)
    data = data.as_matrix()
    DCM = data[:, [0,1,2, 4,5,6, 8,9,10]]
    pos = data[:,[3, 7, 11]]
    vel = pos2vel(pos)

    DCM = DCM[1:,:]
    pos = pos[1:,:]
    vel = vel[1:,:]
    return DCM, pos, pos2vel(pos)

def readPred(num):
    dir = 'Results/Pred_Data/seq'
    velName = dir + str(num) + '_vel.txt'
    posName = dir + str(num) + '_pos.txt'
    vel_pr = pd.read_csv(velName, sep=" ", header=None).as_matrix()
    pos_pr = vel2pos(vel_pr)
    return vel_pr, pos_pr

def getMergedData(seqList):
    velList_gt = None
    posList_gt = None
    velList_pr = None
    posList_pr = None

    for seq in seqList:
        DCM, pos, vel = readGT(seq)
        velList_gt = vel if velList_gt is None else np.concatenate([velList_gt, vel], axis=0)
        posList_gt = pos if posList_gt is None else np.concatenate([posList_gt, pos], axis=0)

        vel_p, pos_p = readPred(seq)
        velList_pr = vel_p if velList_pr is None else np.concatenate([velList_pr, vel_p], axis=0)
        posList_pr = pos_p if posList_pr is None else np.concatenate([posList_pr, pos_p], axis=0)

    return velList_gt, posList_gt, velList_pr, posList_pr

def getSeqInput(x,y, T):
    input = None
    label = None
    offset = T-1
    zeropads = np.zeros((offset, x.shape[1]))
    #x = np.concatenate([zeropads, x], axis=0)
    for i in range(0, x.shape[0]-offset):
        temp = np.reshape(x[i:i+T, :], (1, -1, 6))
        temp2 = np.reshape(y[i:i+T, :], (1, -1, 6))
        input = temp if input is None else np.concatenate([input, temp], axis=0)
        label = temp2 if label is None else np.concatenate([label, temp2], axis=0)
    # label = y[0:y.shape[0],:]
    # label = np.concatenate([np.zeros((label.shape[0],3)), label], axis=1)
    return input, label

def getSeqData(seq, T):
    DCM, pos, vel = readGT(seq)
    vel_p, pos_p = readPred(seq)
    seq_input, seq_label = getSeqInput(vel_p, vel, T)
    return seq_input, seq_label

def getSeqData2(seq, T):
    DCM, pos, vel = readGT(seq)
    vel_p, pos_p = readPred(seq)
    measPos = pos + (np.random.rand(pos.shape[0], 3)-0.5)*2
    pred = np.concatenate([measPos, vel_p], axis=1)
    gt = np.concatenate([pos, vel], axis=1)
    seq_input, seq_label = getSeqInput(pred, gt, T)
    return seq_input, seq_label

def getMergedSeqData(seqList, T):
    x_list = None
    y_list = None

    for seq in seqList:
        print(T)
        x, y = getSeqData2(seq, T)
        print(x.shape)
        x_list = x if x_list is None else np.concatenate([x_list, x], axis=0)
        y_list = y if y_list is None else np.concatenate([y_list, y], axis=0)
    return x_list, y_list

if __name__ == '__main__':
    DCM, pos, vel = readGT(0)
    vel_p, pos_p = readPred(0)
    plt.figure()
    plt.plot(vel_p, 'b')
    plt.plot(vel, 'r.', markersize=1)

    plt.show()

    # print vel_gt.shape
    # print pos_gt.shape
    # print vel_pr.shape
    # print pos_pr.shape









#end
