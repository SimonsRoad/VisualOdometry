import matplotlib.pyplot as plt
from PrepData import *
from Model import *
import time
import sys
import cv2

def runTrain():
    of, vel, pos, DCM, img1, img2 = getMergedData([0])

    m = getCNN(320, 1152)
    m.load_weights('Weights/temp_sg.h5')
    m.fit([img1, img2, DCM], [of, vel], epochs=20, batch_size=10, verbose=1,  shuffle=False)
    m.save_weights('Weights/temp_sg_2.h5')
    print 'done'

def runTest():
    m = getCNN(320, 1152)
    m.load_weights('Weights/temp_mu.h5')
    of, vel, pos, DCM, img1, img2 = getMergedData([5])
    pred_vel_list = []

    i = 0
    while i < img1.shape[0]:
        inputImg1 = img1[i:i+10,:,:,:]
        inputImg2 = img2[i:i+10,:,:,:]
        dcm = DCM[i:i+10,:]
        pred = m.predict([inputImg1, inputImg2, dcm])
        pred_vel = pred[1]
        pred_vel_list.append(pred_vel)
        i += 10
        if i%100 == 0:
            print i

    pred_vel = np.array(pred_vel_list)
    pred_vel = np.reshape(pred_vel, (vel.shape[0], vel.shape[1]))
    diff = vel-pred_vel

    print pred_vel.shape
    print np.mean(diff)
    plt.figure()
    plt.plot(vel, 'r')
    plt.plot(pred_vel, 'b')
    plt.show()


    pred_pos = vel2pos(pred_vel)
    plt.figure()
    plt.plot(pos[:,0], pos[:,2], 'ro')
    plt.plot(pred_pos[:,0], pred_pos[:,2], 'b.')
    plt.show()


if __name__=='__main__':
    type = int(sys.argv[1])
    if type==0:
        runTrain()
    elif type==1:
        runTest()

#end
