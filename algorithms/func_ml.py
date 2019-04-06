import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,ExtraTreesClassifier
from sklearn.model_selection import KFold
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import logging

def train_model_prob(model, x, y, num_class, k=3):
##    min_max_scaler = preprocessing.MinMaxScaler()
##    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
##    ni = np.unique(y)
##    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]   
        model.fit(x_train,y_train)
        y_predict[test_index,:] = model.predict_proba(x_test)
    #print('train shape ', y_predict.shape)
    return y_predict

def train_model(model, x, y, k=3):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    y_predict = np.zeros((len(y)))
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        y_predict[test_index] = model.predict(x_test)
    #print('train shape ', y_predict.shape)
    return y_predict

def svm_train_model(model, x, y, k=3):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(np.asarray(x))
    kf = StratifiedKFold(n_splits=k)
    ni = np.unique(y)
    num_class = ni.shape[0]
    y_predict = np.zeros((len(y), num_class))
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]   
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_label = []
        for i in y_pred:
            binary_label = np.zeros((num_class))
            binary_label[int(i)] = 1
            y_label.append(binary_label)
        y_predict[test_index,:] = np.asarray(y_label)
    return y_predict

def test_function_svm(model, x_train, y_train, num_class, x_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    x_test = min_max_scaler.transform(np.asarray(x_test))
    logging.info('Training set shape in testing '+str(x_train.shape))
    logging.info('Test set shape in testing'+str(x_test.shape))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_label = []
    ni = np.unique(y_train)
    num_class = ni.shape[0]
    for i in y_pred:
        binary_label = np.zeros((num_class))
        binary_label[int(i)] = 1
        y_label.append(binary_label)
    y_predict = np.asarray(y_label)
    return y_predict

def test_function_prob(model, x_train, y_train, x_test):
    logging.info('Training set shape in testing '+str(x_train.shape))
    logging.info('Test set shape in testing'+str(x_test.shape))
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    x_test = min_max_scaler.transform(np.asarray(x_test))
    model.fit(x_train, y_train)
    print(model, x_train.shape, y_train.shape, x_test.shape)
    y_pred = model.predict_proba(x_test)
    #print('test shape', y_pred.shape)
    return y_pred

def test_function(model, x_train, y_train, x_test):
    logging.info('Training set shape in testing '+str(x_train.shape))
    logging.info('Test set shape in testing'+str(x_test.shape))
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(np.asarray(x_train))
    x_test = min_max_scaler.transform(np.asarray(x_test))
    model.fit(x_train, y_train)
    #print(model, x_train.shape, y_train.shape, x_test.shape)
    #print(model.score(x_train,y_train))
    y_pred = model.predict(x_test)
    return y_pred

def linear_svm(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = LinearSVC()
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def rbf_svm(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = SVC(kernel = 'rbf')
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def knn(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = KNeighborsClassifier(n_neighbors=1)
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def lr(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = LogisticRegression()
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def mlp(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = MLPClassifier()
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def nb(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = GaussianNB()
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def randomforest(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = RandomForestClassifier(n_estimators=500, max_depth=100)
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def adb(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100), n_estimators=500)
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def erandomforest(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = ExtraTreesClassifier(n_estimators=500, max_depth=100)
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def xgbc(x_train, y_train, num_train):
    #print(x_train.shape, y_train.shape, num_train, x_train[0:num_train,:].shape, x_train[num_train:-1,:].shape)
    classifier = xgb.XGBClassifier(n_estimators=500, max_depth=100, learning_rate=1)
    if num_train==x_train.shape[0]:
        y_labels = train_model(classifier, x_train, y_train)
    else:
        y_labels = test_function(classifier, x_train[0:num_train,:], y_train, x_train[num_train:x_train.shape[0],:])
    return y_labels

def voting_vector(*args):
    #print(args)
    x = Counter(args)
    #print(x.most_common(1)[0][0])
    return x.most_common(1)[0][0]

def voting_two(cl1, cl2):
    label = []
    for i in range(cl1.shape[0]):
        label.append(voting_vector(cl1[i], cl2[i]))
    return np.asarray(label)

def voting_three(cl1, cl2, cl3):
    label = []
    for i in range(cl1.shape[0]):
        label.append(voting_vector(cl1[i], cl2[i], cl3[i]))
    return np.asarray(label)

def voting_five(cl1, cl2, cl3, cl4, cl5):
##    print(cl1, cl2, cl3, cl4, cl5)
    label = []
    for i in range(cl1.shape[0]):
        label.append(voting_vector(cl1[i], cl2[i], cl3[i], cl4[i], cl5[i]))
    return np.asarray(label)

def voting_seven(cl1, cl2, cl3, cl4, cl5, cl6, cl7):
    label = []
    for i in range(cl1.shape[0]):
        label.append(voting_vector(cl1[i], cl2[i], cl3[i], cl4[i], cl5[i], cl6[i], cl7[i]))
    return np.asarray(label)

def average_c(*args):
    if len(args) == 1:
        return args
    else:
        average_prob = np.mean(np.asarray(args), axis=0)
    #print('arveg_c', average_prob)
    return average_prob

###not finish the average weighting, there are problem about the test set.
def average_weightings(y_train, *args):
    if len(args) == 1:
        return args
    else:
        acc = []
        for i in range(0,len(args)):
            y_predict = np.argmax(args[i], axis=1)
            acc.append(np.sum(y_predict == y_train) / len(y_train))
        acc = np.asarray(acc)
        acc_all = acc/np.sum(acc)
        average_prob = np.zeros(args[0].shape)
        for i in range(0, len(acc_all)):
            average_prob= average_prob+acc_all[i]*args[i]
    #print('arveg', average_prob)
    return average_prob

if __name__ == "__main__":
    data_set = 'fashion_mnist'
    file_path = '/vuw/code/datasets_npy/popular/'
    file_name = file_path+data_set+'/'+data_set
    #load data
    x_train = np.load(file_name+'_train_data.npy')/255
    y_train = np.load(file_name+'_train_label.npy')
    x_eval2, x_grow, y_eval2, y_grow = train_test_split(x_train, y_train, test_size=0.005, random_state=5)
    x_eval2, x_grow2, y_eval2, y_grow2= train_test_split(x_train, y_train, test_size=0.006, random_state=5)

    x_grow = np.reshape(x_grow, (x_grow.shape[0], x_grow.shape[1]*x_grow.shape[2]))
    x_grow2 = np.reshape(x_grow2, (x_grow2.shape[0], x_grow2.shape[1]*x_grow2.shape[2]))
    print(x_grow.shape, x_grow2.shape)
    print('lsvm',linear_svm(x_grow,y_grow,x_grow2).shape)
    print('svc',rbf_svm(x_grow,y_grow,x_grow2).shape)
    print('knn',knn(x_grow,y_grow,x_grow2).shape)
    print('mlp',mlp(x_grow,y_grow,x_grow2).shape)
    print('rf',randomforest(x_grow,y_grow,x_grow2).shape)
    print('adb',adb(x_grow,y_grow,x_grow2).shape)
    print('erf',erandomforest(x_grow,y_grow,x_grow2).shape)
