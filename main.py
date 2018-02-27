import matplotlib.pyplot as plt
from PrepData import *
from Model import *
import time
import sys
import cv2
from keras.callbacks import EarlyStopping
import pickle

def runTrainCNN():
    fn = getModel(360, 640)
    fn.load_weights('Weights/b3_evenlight.h5')
    of, vel, pos, DCM, img1, img2 = getMergedData([0,2 6, 4])
    earlystop = EarlyStopping(monitor='loss', min_delta=10**-5, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    history = fn.fit([img1, img2, DCM], vel, epochs=10, batch_size=32, verbose=1,  shuffle=True, callbacks=callbacks_list)
    fn.save_weights('Weights/b3_evenlight.h5')
    print 'done'


def runTest():
    m = getModel(360, 640)
    m.load_weights('Weights/b3_evenlight.h5')
    #runTestSeq(m,6)
    for seq in range(8,11):
        runTestSeq(m,seq)


def runTestSeq(m,seq):
    of, vel, pos, DCM, img1, img2 = getMergedData([seq])
    pred_vel_list = []
    i = 0
    while i < img1.shape[0]:
        inputImg1 = img1[i:i+10,:,:,:]
        inputImg2 = img2[i:i+10,:,:,:]
        dcm = DCM[i:i+10,:]
        pred = m.predict([inputImg1, inputImg2, dcm])
        pred_vel_list.append(pred)
        i += 10
        if i%100 == 0:
            print i

    pred_vel = np.array(pred_vel_list)
    print pred_vel.shape
    pred_vel = np.reshape(pred_vel, (vel.shape[0], 3))
    #pred_vel = pred_vel[:,3:]
    diff = vel-pred_vel

    print pred_vel.shape
    print np.mean(diff)

    pred_pos = pos2vel(pred_vel)
    rmse_vel = np.sqrt((np.asarray((np.subtract(pred_vel, vel))) ** 2).mean())
    print 'rmse=%f' %(rmse_vel)
    rmse_pos = np.sqrt((np.asarray((np.subtract(pred_pos, pos))) ** 2).mean())
    print 'rmse=%f' %(rmse_pos)

    np.savetxt('Results/Pred_Data/seq'+str(seq)+'_vel.txt', pred_vel)
    np.savetxt('Results/Pred_Data/seq'+str(seq)+'_pos.txt', pred_pos)

    fig = plt.figure()
    plt.plot(vel, 'r')
    plt.plot(pred_vel, 'b')
    #plt.show()
    fig.savefig('Results/Images/seq' + str(seq) + '_vel.png')


    pred_pos = vel2pos(pred_vel)
    fig = plt.figure()
    plt.plot(pos[:,0], pos[:,2], 'ro')
    plt.plot(pred_pos[:,0], pred_pos[:,2], 'b.')
    if seq == 4:
        plt.xlim((-200, 200))
    if seq == 6:
        plt.xlim((-50, 50))
    #plt.show()
    fig.savefig('Results/Images/seq' + str(seq) + '_pos.png')

if __name__=='__main__':
    type = int(sys.argv[1])
    if type==0:
        runTrainCNN()
    elif type==1:
        runTest()

#end
