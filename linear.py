from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

def make_matrix_to_p(matrix):
    N=int(input('Degree of polynomial = '))
    poly=PolynomialFeatures(degree=N)
    X_poly=poly.fit_transform(matrix)
    return(X_poly)

def Primal_Form(X,y):
    try:
        P=make_matrix_to_p(X)
        a=np.linalg.inv(P.transpose().dot(P))
        w=a.dot(P.transpose())
        W=w.dot(y)
        return W
    except np.linalg.LinAlgError:
        return('It is not in Primal Form maybe DUAL FORM')


def Dual_Form(X,y):
    try:
        P=make_matrix_to_p(X)
        a=np.linalg.inv(P.dot(P.transpose()))
        w=P.transpose().dot(a)
        W=w.dot(y)
        return W
    except np.linalg.LinAlgError:
        return('It is not in Dual Form maybe PRIMAL FORM')

def predict_using_p(X,y,Xtest):
    X_Ptest=make_matrix_to_p(Xtest)
    if type(Primal_Form(X,y))!=str:
        W=Primal_Form(X,y)
        y_test=X_Ptest.dot(W)
        print('Predicted value solved by Primal Form is')
        return y_test
    else:
        W=Dual_Form(X,y)
        y_test=X_Ptest.dot(W)
        print('Predicted value solved by Dual Form is')
        return y_test

def OD(X,y):
    a=np.linalg.inv(X.transpose().dot(X))
    w=a.dot(X.transpose())
    W=w.dot(y)
    return W

def UD(X,y):
    a=np.linalg.inv(X.dot(X.transpose()))
    w=X.transpose().dot(a)
    W=w.dot(y)
    return W

def predict_using_OD(X,y,Xtest):
    print('Make sure that you HAVE added OFFSET!!!')
    W=OD(X,y)
    ytest=Xtest.dot(W)
    print('Predicted value is')
    return ytest

def predict_using_UD(X,y,Xtest):
    print('Make sure that you HAVE added OFFSET!!!')
    W=UD(X,y)
    ytest=Xtest.dot(W)
    print('Predicted value is')
    return ytest

def binary_classification(X,y,xtest):
    W=OD(X,y)
    ytest=xtest.dot(W)
    print(ytest)
    print(f'the signum value is {1 if ytest[0]>0 else -1}')

def make_onehot(matrix):
    onehot_encoder=OneHotEncoder(sparse=False)
    reshaped=matrix.reshape(len(matrix), 1)
    y_onehot=onehot_encoder.fit_transform(reshaped)
    return y_onehot

def multiclass_using_linear(X,y,Xtest):
    W=OD(X,y)
    ytest=xtest.dot(W)
    print('Predicted value is')
    print(ytest)
    print('Predicted class using onehot is')
    lst=list(ytest)
    print(lst.index(max(lst)))

def multiclass_using_p(X,y,Xtest):
    poly=PolynomialFeatures(int(input('Degree of polynomial = ')))
    xtest=poly.fit_transform(Xtest)
    if type(Primal_Form(X,y))!=str:
        W=Primal_Form(X,y)
    else:
        W=Dual_Form(X,y)
    print(xtest.dot(W))
    print('Predicted value is ')
    lst=xtest.dot(W).tolist()
    print(lst[0].index(max(lst[0])))
    








