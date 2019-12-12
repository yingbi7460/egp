#python packages
import random
import time
import operator
import evalGP
import sys
# only for strongly typed GP
import gp_restrict
import numpy
# deap package
from deap import base, creator, tools, gp
# fitness function
from FEVal_new import evalTrain
from FEVal_new import evalTest_fromvector as evalTest
from strongGPDataType import numInts
from strongGPDataType import kernelSize,histdata,filterData,coordsX1,coordsX2,trainData
from strongGPDataType import windowSize2,windowSize3,poolingType, imageDa, trainLabel
# defined by author
import functionSet as fs
import func_ml as fs_ml
from sklearn.model_selection import train_test_split

randomSeeds=int(sys.argv[2])
dataSetName=str(sys.argv[1])

def load_data(dataset_name, path=None):
    if path is not None:
        file = path+dataset_name+'/'+dataset_name
    else: file = dataset_name
    x_train = numpy.load(file+'_train_data.npy')/255.0
    y_train = numpy.load(file+'_train_label.npy')
    x_test = numpy.load(file+'_test_data.npy')/255.0
    y_test = numpy.load(file+'_test_label.npy')
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data(dataSetName)
print(x_train.shape,y_train.shape, x_test.shape,y_test.shape)

#parameters:
num_train = x_train.shape[0]
population=100
generation=50
cxProb=0.8
mutProb=0.19
elitismProb=0.01
totalRuns = 1
initialMinDepth=2
initialMaxDepth=8
maxDepth=8

##GP
pset = gp.PrimitiveSetTyped('MAIN', [filterData, trainLabel], trainData, prefix = 'Image')
##imageDa
pset.addPrimitive(fs_ml.voting_two, [trainData, trainData, trainData], trainData, name='Voting23')
pset.addPrimitive(fs_ml.voting_three, [imageDa, imageDa, imageDa], trainData, name='Voting3')
pset.addPrimitive(fs_ml.voting_five, [imageDa, imageDa, imageDa, imageDa, imageDa], trainData, name='Voting5')
pset.addPrimitive(fs_ml.voting_seven, [imageDa, imageDa, imageDa, imageDa, imageDa, imageDa, imageDa], trainData, name='Voting7')

#learn classifiers
##pset.addPrimitive(fs_ml.rbf_svm, [histdata, trainLabel, numInts], imageDa, name='SVM_rbf')
pset.addPrimitive(fs_ml.linear_svm, [histdata, trainLabel, numInts], imageDa, name='SVM_linear')
pset.addPrimitive(fs_ml.knn, [histdata, trainLabel, numInts], imageDa, name='KNN')
##pset.addPrimitive(fs_ml.mlp, [histdata, trainLabel, numInts], imageDa, name='MLP')
pset.addPrimitive(fs_ml.lr, [histdata, trainLabel, numInts], imageDa, name='LR')
pset.addPrimitive(fs_ml.randomforest, [histdata, trainLabel, numInts], imageDa, name='RF')
pset.addPrimitive(fs_ml.adb, [histdata, trainLabel, numInts], imageDa, name='ADB')
pset.addPrimitive(fs_ml.erandomforest, [histdata, trainLabel, numInts], imageDa, name='ERF')
###learned features
pset.addPrimitive(fs.FeaCon2, [histdata, histdata], histdata, name ='Root2')
pset.addPrimitive(fs.root_conVector2, [poolingType, poolingType], histdata, name ='Conver2')
pset.addPrimitive(fs.root_conVector3, [poolingType, poolingType, poolingType], histdata, name ='Conver3')
##with other features
#pooling
pset.addPrimitive(fs.maxP, [poolingType, kernelSize, kernelSize], poolingType, name='MaxPf')
pset.addPrimitive(fs.maxP, [filterData, kernelSize, kernelSize], poolingType,name='MaxP1')
#aggregation
pset.addPrimitive(fs.mixconadd, [filterData, float, filterData, float], filterData, name='Mix_ConAdd')
pset.addPrimitive(fs.mixconsub, [filterData, float, filterData, float], filterData, name='Mix_ConSub')
pset.addPrimitive(fs.sqrt, [filterData], filterData, name='Sqrt')
pset.addPrimitive(fs.relu, [filterData], filterData, name='Relu')
# edge features
pset.addPrimitive(fs.sobelxy, [filterData], filterData, name='Sobel_XY')
pset.addPrimitive(fs.sobelx, [filterData], filterData, name='Sobel_X')
pset.addPrimitive(fs.sobely, [filterData], filterData, name='Sobel_Y')
#Gabor
pset.addPrimitive(fs.gab, [filterData, windowSize2, windowSize3], filterData, name='Gabor2')
pset.addPrimitive(fs.gaussian_Laplace1, [filterData], filterData, name='LoG1')
pset.addPrimitive(fs.gaussian_Laplace2, [filterData], filterData, name='LoG2')
pset.addPrimitive(fs.laplace, [filterData], filterData, name='Lap')
pset.addPrimitive(fs.lbp, [filterData], filterData, name='LBP')
pset.addPrimitive(fs.hog_feature, [filterData], filterData, name='HoG')
# Gaussian features
pset.addPrimitive(fs.gau, [filterData, coordsX2], filterData, name='Gau2')
pset.addPrimitive(fs.gauD, [filterData, coordsX2, coordsX1, coordsX1], filterData, name='Gau_D2')
#general filters
pset.addPrimitive(fs.medianf, [filterData], filterData,name='Med')
pset.addPrimitive(fs.maxf, [filterData], filterData,name='Max')
pset.addPrimitive(fs.minf, [filterData], filterData,name='Min')
pset.addPrimitive(fs.meanf, [filterData], filterData,name='Mean')
##pset.addPrimitive(fs.regionS,[region,indexType1,indexType2,windowSize],ndarray,name='Region_S1')
##pset.addPrimitive(fs.regionR,[region,indexType1,indexType2,windowSize,windowSize],ndarray,name='Region_R1')
#Terminals
pset.renameArguments(ARG0='grey')
##pset.addEphemeralConstant('index1',lambda: random.randint(0,bound1-20),indexType1)
pset.addEphemeralConstant('Num_Int',lambda: random.randint(num_train,num_train),numInts)
pset.addEphemeralConstant('randomD',lambda:round(random.random(),3),float)
pset.addEphemeralConstant('kernelSize',lambda:random.randrange(2,4,2),kernelSize)
pset.addEphemeralConstant('Theta',lambda:random.randint(0,8),windowSize2)
pset.addEphemeralConstant('Frequency',lambda:random.randint(0,5),windowSize3)
pset.addEphemeralConstant('Singma',lambda:random.randint(1,4),coordsX2)
pset.addEphemeralConstant('Order',lambda:random.randint(0,3),coordsX1)
##
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#genetic operator
toolbox.register("evaluate", evalTrain,toolbox, trainData=x_train,trainLabel=y_train)
toolbox.register("select", tools.selTournament,tournsize=7)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))

def GPMain(randomSeeds):

    random.seed(randomSeeds)
   
    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    #stats_size_feature = tools.Statistics(key= lambda ind: feature_length(ind, x_train[1,:,:], toolbox))
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                    stats=mstats, halloffame=hof, verbose=True)

    return pop,log, hof

if __name__ == "__main__":
    beginTime = time.clock()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.clock()
    trainTime = endTime - beginTime

    testResults = evalTest(toolbox, hof[0], x_train, y_train,x_test, y_test)
    #saveFile.saveLog(str(randomSeeds) + 'all_pop.pickle', pop)
    #saveFile.saveLog(str(randomSeeds) + 'best_pop.pickle', hof)

    testTime = time.clock() - endTime
    print('testResults ', testResults)
    
    # print(train_tf.shape, test_tf.shape)
    num_features = 0
    #saveFile.saveAllResults(randomSeeds, dataSetName, hof, log,
                                #hof, num_features, trainTime, testTime, testResults)
