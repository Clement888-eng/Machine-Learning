import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def Decision_Tree(N, TestSize, MaxTreeDepth):
    random_state = N
    X,y=fetch_california_housing().data, fetch_california_housing().target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TestSize,random_state=N)

    ytr_Tree_list=[]
    for i in range(1,MaxTreeDepth+1):
        regressor=DecisionTreeRegressor(criterion='mse', max_depth=i, random_state=0)
        regressor.fit(X_train,y_train)
        ytr_Tree_list.append(regressor.predict(X_train))
        
    yts_Tree_list=[]
    for i in range(1,MaxTreeDepth+1):
        regressor=DecisionTreeRegressor(criterion='mse', max_depth=i, random_state=0)
        regressor.fit(X_train,y_train)
        yts_Tree_list.append(regressor.predict(X_test))
        
    mse_trainTree_array=[]
    for i in range(len(ytr_Tree_list)):
        mse_trainTree_array.append(mean_squared_error(ytr_Tree_list[i],y_train))
    mse_trainTree_array=np.array(mse_trainTree_array)
    
    mse_testTree_array=[]
    for i in range(len(yts_Tree_list)):
        mse_testTree_array.append(mean_squared_error(yts_Tree_list[i],y_test))
    mse_testTree_array=np.array(mse_testTree_array)

    ytr_Forest_list=[]
    for i in range(1,MaxTreeDepth+1):
        regressor=RandomForestRegressor(criterion='mse', max_depth=i, random_state=0)
        regressor.fit(X_train,y_train)
        ytr_Forest_list.append(regressor.predict(X_train))

    yts_Forest_list=[]
    for i in range(1,MaxTreeDepth+1):
        regressor=RandomForestRegressor(criterion='mse', max_depth=i, random_state=0)
        regressor.fit(X_train,y_train)
        yts_Forest_list.append(regressor.predict(X_test))

    mse_trainForest_array=[]
    for i in range(len(ytr_Forest_list)):
        mse_trainForest_array.append(mean_squared_error(ytr_Forest_list[i],y_train))
    mse_trainForest_array=np.array(mse_trainForest_array)
    
    mse_testForest_array=[]
    for i in range(len(yts_Forest_list)):
        mse_testForest_array.append(mean_squared_error(yts_Forest_list[i],y_test))
    mse_testForest_array=np.array(mse_testForest_array)

    

    
    
        
    """
    Input type
    :N type: int
    :TestSize type: float
    :MaxTreeDepth type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 8)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 8)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :ytr_Tree_list type: List[numpy.ndarray]
    :yts_Tree_list type: List[numpy.ndarray]
    :mse_trainTree_array type: numpy.ndarray
    :mse_testTree_array type: numpy.ndarray
    :ytr_Forest_list type: List[numpy.ndarray]
    :yts_Forest_list type: List[numpy.ndarray]
    :mse_trainForest_array type: numpy.ndarray
    :mse_testForest_array type: numpy.ndarray
    """
    # your code goes here


    # return in this order
    return X_train, y_train, X_test, y_test, ytr_Tree_list, yts_Tree_list, mse_trainTree_array, mse_testTree_array, ytr_Forest_list, yts_Forest_list, mse_trainForest_array, mse_testForest_array

