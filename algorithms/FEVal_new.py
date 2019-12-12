import numpy as np

def evalTrain(toolbox, individual, hof, trainData, trainLabel):
##    print(individual)
    if len(hof) != 0 and individual in hof:
        ind = 0
        while ind < len(hof):
            if individual == hof[ind]:
                accuracy, = hof[ind].fitness.values
                ind = len(hof)
            else: ind+=1
    else:
        try:
            func = toolbox.compile(expr=individual)
            output = np.asarray(func(trainData, trainLabel))
##            print(train_tf.shape)
##            print(individual)
            accuracy = 100*np.sum(output == trainLabel) / len(trainLabel)
        except:
            accuracy=0
##            print(individual)
    #print(accuracy)
    return accuracy,

def evalTest_fromvector(toolbox, individual, trainData, trainLabel, test, testL):
    x_train = np.concatenate((trainData, test), axis=0)
    func = toolbox.compile(expr=individual)
    output = np.asarray(func(x_train, trainLabel))
##    print(output[0:10])
    accuracy = 100*np.sum(output==testL)/len(testL)
##    print(np.asarray(train_tf).shape, np.asarray(test_tf).shape)    
    return accuracy

