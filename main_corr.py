from PrepData_post import *
import matplotlib.pyplot as plt
import numpy as np
from Model_PosQ import *
import sys

def getSeqInput(x,y, T):
    input = None
    offset = T-1
    zeropads = np.zeros((offset, x.shape[1]))
    x = np.concatenate([zeropads, x], axis=0)
    for i in range(0, x.shape[0]-offset):
        temp = np.reshape(x[i:i+T, :], (1, -1,6))
        input = temp if input is None else np.concatenate([input, temp], axis=0)
    label = y[0:y.shape[0],:]
    label = np.concatenate([np.zeros((label.shape[0],3)), label], axis=1)
    return input, label

def main():
    T = 20
    vel_gt, pos_gt, vel_pr, pos_pr = getMergedData([0, 2, 4, 6])
    measPos = pos_gt + (np.random.rand(pos_gt.shape[0], 3)-0.5)*2
    predPos = (np.concatenate([np.zeros((1,3)), measPos], axis=0))[0:pos_gt.shape[0]] + vel_pr

    x = np.concatenate([predPos, measPos], axis=1)/1000
    y = pos_gt/1000
    input, label = getSeqInput(x,y, T)

    model = getPosQ(input.shape[1:], T)
    model.load_weights('Weights/corrPosQ.h5')

    history = model.fit(input, label, epochs=50, batch_size=1000, verbose=1, shuffle=True, validation_split=0.01)
    loss_history = history.history["loss"]

    #model.save_weights('corrPosQ.h5')
    # plt.figure()
    # plt.plot(loss_history)
    # plt.show()
    #model.load_weights('corrPosQ.h5')
def test_quick():
    T = 20
    model = None
    for seq in range(0,1):
        if model is None:
            model = getPosQ((20,6), T)
            model.load_weights('Weights/corrPosQ.h5')

        vel_gt, pos_gt, vel_pr, pos_pr = getMergedData([0])
        measPos = pos_gt + (np.random.rand(pos_gt.shape[0], 3)-0.5)*2
        predPos = (np.concatenate([np.zeros((1,3)), measPos], axis=0))[0:pos_gt.shape[0]] + vel_pr

        x = np.concatenate([predPos, measPos], axis=1)/1000
        y = pos_gt/1000
        input, label = getSeqInput(x,y, T)
        finalPos = model.predict(input)*1000

        plt.close('all')
        plt.figure()
        gt, = plt.plot(pos_gt[:,0], pos_gt[:,2], 'ro')
        pr, = plt.plot(pos_pr[:,0], pos_pr[:,2], 'b')
        meas, = plt.plot(measPos[:,0], measPos[:,2], 'cyan')
        corr, = plt.plot(finalPos[:,3], finalPos[:,5], 'g')
        plt.legend([gt, pr, meas, corr], ['Ground Truth', 'Predicted', 'Measured', 'Corrected'])
        plt.xlabel('Distance, meters')
        plt.ylabel('Distance, meters')
        plt.show()


def test_real():
    T = 20
    model = None
    for seq in range(0,11):
        if model is None:
            model = getPosQ((20,6), T)
            model.load_weights('Weights/corrPosQ.h5')
        vel_gt, pos_gt, vel_pr, pos_pr = getMergedData([seq])
        measPos = pos_gt+ (np.random.rand(pos_gt.shape[0], 3)-0.5)*2
        pos_delay = np.zeros((20,3))
        measPos_delay = np.zeros((20,3))
        pos_corr = np.zeros((1,6))
        pos_corr_list = np.zeros((pos_pr.shape[0],6))
        for i in range(0, vel_pr.shape[0]):
            pos_pred = pos_corr[:,3:] + vel_pr[None, i,:]
            pos_delay = np.concatenate([pos_delay[1:,:], pos_pred], axis = 0)
            measPos_delay = np.concatenate([measPos_delay[1:,:], measPos[None,i,:]], axis = 0)
            input = np.concatenate([pos_delay, measPos_delay], axis=1)
            input = np.reshape(input/1000, (1, input.shape[0], input.shape[1]))
            pos_corr = model.predict(input)*1000
            pos_corr_list[i,:] = pos_corr
            if i%200 == 0:
                print (i)
        finalPos = pos_corr_list
        print (finalPos.shape)

        plt.close('all')
        plt.figure()
        gt, = plt.plot(pos_gt[:,0], pos_gt[:,2], 'ro')
        pr, = plt.plot(pos_pr[:,0], pos_pr[:,2], 'b')
        meas, = plt.plot(measPos[:,0], measPos[:,2], 'cyan')
        corr, = plt.plot(finalPos[:,3], finalPos[:,5], 'g')
        plt.legend([gt, pr, meas, corr], ['Ground Truth', 'Predicted', 'Measured', 'Corrected'])
        plt.xlabel('Distance, meters')
        plt.ylabel('Distance, meters')
        plt.savefig('Results/Images/seq_corr' + str(seq))

        np.savetxt('Results/corr_data/seq_measPos' + str(seq) + '.txt', measPos)
        np.savetxt('Results/corr_data/seq_corrPos' + str(seq) + '.txt', finalPos)
        #plt.show()


if __name__=='__main__':
    type = int(sys.argv[1])
    if type==0:
        main()
    elif type==1:
        test_quick()
    elif type==2:
        test_real()
