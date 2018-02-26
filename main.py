import matplotlib.pyplot as plt
from PrepData import *
from Model import *
import time
import sys
import cv2
from keras.callbacks import EarlyStopping
import pickle

def runTrainCNN():
    of, vel, pos, DCM, img1, img2 = getMergedData([7])
    fn = getCNN(320, 1152)
    earlystop = EarlyStopping(monitor='loss', min_delta=10**-5, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    history = fn.fit([img1, img2], of, epochs=20, batch_size=10, verbose=1,  shuffle=True, callbacks=callbacks_list)
    fn.save_weights('Weights/b2_light_cnn.h5')
    print 'done'

def runTrainModel():
    m = getModel(320, 1152)
    #m.load_weights('Weights/b2_light_final.h5')
    of, vel, pos, DCM, img1, img2 = getMergedData([7])
    earlystop = EarlyStopping(monitor='dense_6_loss', min_delta=10**-5, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    history = m.fit([img1, img2, DCM], [of, vel], epochs=20, batch_size=10, verbose=1,  shuffle=True, callbacks=callbacks_list)
    m.save_weights('Weights/b2_light_final.h5')
    print 'done'

def runTest():
    m = getModel(320, 1152)
    m.load_weights('Weights/b2_light_final.h5')
    of, vel, pos, DCM, img1, img2 = getMergedData([1])
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
    print pred_vel.shape
    pred_vel = np.reshape(pred_vel, (vel.shape[0], 3))
    #pred_vel = pred_vel[:,3:]
    diff = vel-pred_vel

    print pred_vel.shape
    print np.mean(diff)

    pred_pos = pos2vel(pred_vel)
    rmse = np.sqrt((np.asarray((np.subtract(pred_vel, vel))) ** 2).mean())
    print 'rmse=%f' %(rmse)
    rmse = np.sqrt((np.asarray((np.subtract(pred_pos, pos))) ** 2).mean())
    print 'rmse=%f' %(rmse)


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
        runTrainCNN()
    elif type==1:
        runTrainModel()
    elif type==2:
        runTest()

#end
