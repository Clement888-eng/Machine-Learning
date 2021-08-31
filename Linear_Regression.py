from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error

def Primal_Form(X,y):
    regl=0.0001*np.identity(X.shape[1])
    a=np.linalg.inv(X.transpose().dot(X)+regl)
    w=a.dot(X.transpose())
    W=w.dot(y)
    return W

def Dual_Form(X,y):
    regl=0.0001*np.identity(X.shape[0])
    a=np.linalg.inv(X.dot(X.transpose())+regl)
    w=X.transpose().dot(a)
    W=w.dot(y)
    return W

def Linear(N):
    random_state = N
    X,y=load_iris().data, load_iris().target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=N)
    onehot_encoder=OneHotEncoder(sparse=False)
    reshaped=y_train.reshape(len(y_train), 1)
    y_onehot=onehot_encoder.fit_transform(reshaped)
    Ytr=y_onehot
    onehot_encoder=OneHotEncoder(sparse=False)
    reshaped=y_test.reshape(len(y_test), 1)
    y_onehot=onehot_encoder.fit_transform(reshaped)
    Yts=y_onehot
    Ptrain_list=list()
    for i in range(1,11):
        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X_train)
        Ptrain_list.append(X_poly)
    Ptest_list=list()
    for i in range(1,11):
        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X_test)
        Ptest_list.append(X_poly)
    
    w_list=list()
    for i in Ptrain_list:
        if len(i)>=len(i[0]):
            w_list.append(Primal_Form(i,Ytr))
        else:
            w_list.append(Dual_Form(i,Ytr))
            
    error_train_array=list()
    for i in range(len(w_list)):
        y_pred=Ptrain_list[i].dot(w_list[i])
        yt_cls_p = [[1 if y == max(x) else 0 for y in x] for x in y_pred ]
        m1 = np.matrix(Ytr)

        m2 = np.matrix(yt_cls_p)

        difference = np.abs(m1 - m2)
        correct_p = np.where(~difference.any(axis=1))[0]

        error_train_array.append(len(X_train)-len(correct_p))
    error_train_array=np.array(error_train_array)
        
    error_test_array=list()
    for i in range(len(w_list)):
        y_pred=Ptest_list[i].dot(w_list[i])
        yt_cls_p = [[1 if y == max(x) else 0 for y in x] for x in y_pred ]
        m1 = np.matrix(Yts)

        m2 = np.matrix(yt_cls_p)

        difference = np.abs(m1 - m2)
        correct_p = np.where(~difference.any(axis=1))[0]

        error_test_array.append(len(X_test)-len(correct_p))
    error_test_array=np.array(error_test_array)
        
    return X_train, X_test, y_train, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
