#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from sklearn import ensemble
import pickle

def andrewtegify(whatever, start, end):
    duration = start - end
    max = np.amax(whatever) * abs(duration)
    min = np.amin(whatever) * abs(duration)
    steps = abs(duration / len(whatever))
    axis = np.arange(start, end, steps)
    while len(whatever) != len(axis):
        if len(whatever) > len(axis): 
            #print 'add'
            axis = np.append(axis, (axis[len(axis)-1] + steps))
        elif len(whatever) < len(axis): 
            #print 'remove'
            axis = np.delete(axis, (len(axis) - 1))
        #print str(len(whatever)) + ' ' +  str(len(axis))
        
    totar = 0
    for val in range(len(whatever)):
        whatever[val] = abs(whatever[val])
    if max < 0 and min < 0:
        va = min - max
        tarea = scipy.integrate.simps(whatever, axis)
        tarea = min + tarea
        totar = abs(va) - tarea
    elif max > 0 and min < 0:
        va = max - min
        tarea = scipy.integrate.simps(whatever, axis)
        totar = abs(va) -  tarea
    elif max > 0 and min > 0:
        va = max - min
        tarea = scipy.integrate.simps(whatever, axis)
        tarea = tarea - min
        totar = abs(va) - tarea
    return totar
if True:
    #filenames = ['./data_sample/Subject_2_LAYING.txt', './data_sample/Subject_2_SITTING.txt', './data_sample/Subject_2_STANDING.txt', './data_sample/Subject_2_WALKING.txt', './data_sample/Subject_2_WALKDWN.txt', './data_sample/Subject_2_WALKUPS.txt']
    #filenames = ['./data_sample/laying_all.txt','./data_sample/sitting_all.txt','./data_sample/standing_all.txt','./data_sample/walking_all.txt','./data_sample/walking_up_all.txt','./data_sample/walking_down_all.txt']
    filenames = ['Ema_RUN.txt','Ema_WALK.txt','Ema_STILL.txt','Ema_STILL_ARM.txt']
    #filenames = ['running-a.txt','walking-a.txt','standing-a.txt','standing_arm-a.txt']
    #filenames = ['running-b.txt','walking-b.txt','standing-b.txt','standing_arm-b.txt'];
    labels = ['running', 'walking', 'Standing', 'standing_arm'];

    N = len(filenames)
    WINDOW_SIZE = 100
    STEP = 100

    data_all = []
    data_mean = []
    data_var = []
    data_combined = [0, 0, 0];
    data_mean_combined = [0, 0, 0];
    data_var_combined = [0, 0, 0];

    for i in range(N):
        # Add data from next file
        #timeStamp,packetCounter, AccX,AccY,AccZ, GyrX,GyrY,GyrZ, MagX,MagY,MagZ 
        data_all.append(np.loadtxt(filenames[i],delimiter=',',usecols = (2,3,4)))
        data_combined = np.vstack([data_combined, data_all[i]])

	
        [row, col] = data_all[i].shape
	
        _mean = np.zeros((len(range(0, row, STEP)), col))
        _var = np.zeros((len(range(0, row, STEP)), col))
	
        for i_col in range(col):
		
            idx = 0
		
            for i_row in range(0, row, STEP):
	
                # compute mean of window elements
                #data_mean[j:j+WINDOW_SIZE-1,i] = np.sum(data[j:j+WINDOW_SIZE-1,i]) / WINDOW_SIZE
                #_mean[idx,i_col] = np.mean(data_all[i][i_row:i_row+WINDOW_SIZE-1,i_col])
                #_var[idx, i_col] = np.std(data_all[i][i_row:i_row+WINDOW_SIZE-1,i_col])
                _var[idx, i_col] = andrewtegify(data_all[i][i_row:i_row + WINDOW_SIZE-1, i_col], i_row, i_row + WINDOW_SIZE-1)
                _mean[idx, i_col] = np.mean(data_all[i][i_row:i_row + WINDOW_SIZE-1, i_col])
                idx += 1
            # end for i_row
		
        # end for i_col
		
        data_mean.append(_mean)
        data_mean_combined = np.vstack([data_mean_combined, data_mean[i]])
	
        data_var.append(_var)
        data_var_combined = np.vstack([data_var_combined, data_var[i]])
    # end for i

    #np.save('data_mean', data_mean)
    #np.save('data_var', data_var)

    #data_combined = data_combined[1:-1,:]
    #data_mean_combined = data_mean_combined[1:-1,:]
    #data_var_combined = data_var_combined[1:-1,:]

    #plt.figure()
    #plt.plot(data_combined)

    #plt.figure()
    #plt.plot(data_mean_combined)

    #plt.figure()
    #plt.plot(data_var_combined)
    
    nClass = len(data_mean)

    labels = []
    nFeatures = []

    for i in range(nClass):
    
        # labels for the different activities	
        # labels.append(np.ones((len(data_mean[i]),1)) * (i+1))
	
        nFeatures.append(len(data_mean[i]))
	
    # end for i

    N = min(nFeatures)
    
    nTrain = N / 5

    data_rnd = []
    dataTrain = [0, 0, 0, 0, 0, 0]
    labelsTrain = [0]
    dataTest = [0, 0, 0, 0, 0, 0]
    labelsTest = [0]

    for i in range(nClass):
	
        idx = np.arange(N)
        np.random.seed(13)
        np.random.shuffle(idx)

        tmp = np.hstack([data_mean[i][0:N,:], data_var[i][0:N,:]])
        data_rnd.append(tmp[idx])

        labels.append(np.ones((N)) * (i + 1))

        dataTrain = np.vstack([dataTrain, data_rnd[i][0:nTrain,:]])
        #labelsTrain = np.vstack([labelsTrain, labels[i][0:nTrain]])
        labelsTrain = np.append(labelsTrain, labels[i][0:nTrain])
	
        dataTest = np.vstack([dataTest, data_rnd[i][nTrain + 1:-1,:]])
        #labelsTest = np.vstack([labelsTest, labels[i][nTrain+1:-1]])
        labelsTest = np.append(labelsTest, labels[i][nTrain + 1:-1])
	
    # end for i

    dataTrain = dataTrain[1:-1,:]
    labelsTrain = labelsTrain[1:-1]
    dataTest = dataTest[1:-1,:]
    labelsTest = labelsTest[1:-1]

    # create figure and plot data
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(dataTrain)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(labelsTrain)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(dataTest)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(labelsTest)

    clf = ensemble.RandomForestClassifier()

    # Train
    f = open('clfs/test' + str(WINDOW_SIZE) + '-' + str(STEP) + '.clf')
    clf = pickle.load(f)
    labelsPredict = clf.predict(dataTest)

    score = clf.score(dataTest, labelsTest)
    print 'Accuracy %f' % score
    #f.write(str(ws) + ' ' + str(score) + '\n')
plt.figure()
plt.plot(range(len(labelsTest)), labelsTest, range(len(labelsPredict)), labelsPredict)
plt.show()

