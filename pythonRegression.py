import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import os

os.chdir('C:\\Users\\User\\Desktop\\pyscripts')


#Get predictions from a matrix of observations and a given weight matrix

def getPred(x,W):
    return(np.matmul(x,W))


#Compute square loss

def Loss(y,ypred):
    l=(y-ypred)**2
    return(l.sum())


#Compute mean Square Error

def MSE(X,Y,W):
    return((1/X.shape[0])*sum((Y-np.matmul(X,W))**2))


#Store MSE
global cacheMSE
global cacheWeights

cacheMSE= {}
cacheWeights= {}

# Function for calculating gradient descent with shrinkage (reg)

def GradDesc(X,Y,learnRate=0.01,epochs=2000,reg=0):
    
    global cacheLoss
    cacheLoss=[None]*epochs
    
    Weights=np.random.rand(X.shape[1])
    
    Weights=np.array(Weights)
    Weights=Weights.reshape(-1,1)
    m=X.shape[0]
    
    for i in range(epochs):
        
        predictions=getPred(X,Weights)
        cacheLoss[i]=Loss(Y,predictions)
        
        Weights[0]=Weights[0]-(1/m)*learnRate*(np.matmul(X[:,0].transpose(),predictions-Y))
        
        for j in range(1,len(Weights)):
            
            Weights[j]=Weights[j]-(1/m)*learnRate*(np.matmul(X[:,j].transpose(),predictions-Y)+sum(np.dot(Weights[j],reg)))



    return(Weights)


# Load and wrangle


cancerData=pd.read_csv('prostate.txt',delimiter='\t')

trainCancer=cancerData[cancerData.loc[:,'train']=='T']

testCancer=cancerData[cancerData.loc[:,'train']=='F']

x_train=trainCancer.drop(columns=['id','lpsa','train'])
y_train=trainCancer.loc[:,'lpsa']

x_test= testCancer.drop(columns=['id','lpsa','train'])
y_test=testCancer.loc[:,'lpsa']

# Scale predictors

x_train_scaled=sklearn.preprocessing.scale(x_train, axis=0, with_mean=True, with_std=True, copy=True)

x_test_scaled=sklearn.preprocessing.scale(x_test, axis=0, with_mean=True, with_std=True, copy=True)

# Turn into numpy arrays with appropriate shape


x_train_scaled=np.array(x_train_scaled)
y_train=np.array(y_train)
y_train=y_train.reshape(-1,1)

y_test=np.array(y_test)
y_test=y_test.reshape(-1,1)

# Add a column of ones to represent the bias terms


addBias=np.ones([x_train_scaled.shape[0],1])

x_train_scaled=np.append(addBias,x_train_scaled,axis=1)

addBias=np.ones([x_test_scaled.shape[0],1])
x_test_scaled=np.append(addBias,x_test_scaled,axis=1)



# LEAST SQUARES

Wlinear=GradDesc(x_train_scaled,y_train)


LinearMSE=MSE(x_test_scaled,y_test,Wlinear)

# Form Train / Validation set to tune hyperparamters

X_train, X_Validate, Y_train, Y_Validate = sklearn.model_selection.train_test_split( x_train_scaled, y_train, test_size=0.33, random_state=42)


# Note for Lasso and Elastic, since we use built in scikit learn, we don't use the additional bias 1 terms since this is done
#for us in the algorithm. Hence we work with X_train[:,1:] and X_Validate[:,1:] which is everything but the first column

# Find best lambda for ridge

def ChooseLambdaRidge(x,y):
    bestMSE=10e100
    
    lamList=[l*0.05 for l in range(0,300)]

    global Wridge
    global ridgeLambda
    
    for l in lamList:
        Wr=GradDesc(x,y,reg=l)
        if MSE(X_Validate,Y_Validate,Wr)< bestMSE:
            bestMSE=MSE(X_Validate,Y_Validate,Wr)
            ridgeLambda=l
            Wridge=Wr
            cacheMSE['Ridge']=bestMSE
            cacheWeights['Ridge']=Wridge
    return(Wridge)

print(f'The ideal lambda value to use is {ridgeLambda}')

# Get ridge weights and calculate test MSE

Wridge=ChooseLambdaRidge(X_train,Y_train) 

RidgeMSE=MSE(x_test_scaled,y_test,Wridge)



# Get ideal lambda for Lasso


def ChooseLambdaLasso(x,y):
    bestMSE=10e100
    
    alphaList=[l*0.05 for l in range(1,200)]
    
    global bestLassoWeights
    
    for a in alphaList:
        lassoModel=sklearn.linear_model.Lasso(alpha=a,max_iter=3000)
        lassoModel.fit(x,y)
        getPred=lassoModel.predict(X_Validate[:,1:]).reshape(-1,1)
        
        MSE=sum((Y_Validate-getPred)**2)
        if MSE< bestMSE:
            bestMSE=MSE
            bestAlpha=a
            bestLassoInt=lassoModel.intercept_
            bestLassoCoef=lassoModel.coef_
            bestLassoWeights=np.concatenate((bestLassoInt,bestLassoCoef)).reshape(-1,1)
            cacheWeights['Lasso']=bestLassoWeights

    return(bestLassoWeights)




def ChooseParametersElasticNet(x,y):
    bestMSE=10e100
    
    regList=[l*0.02 for l in range(0,700)]
    ratio=[i*0.025 for i in range(0,41)]
   
    global bestElasticInt
    global bestElasticCoef
    global bestAlpha
    global bestRatio
    global bestElasticWeights
    
    for l1 in regList:
        for r in ratio:
            elasticModel=sklearn.linear_model.ElasticNet(alpha=l1,l1_ratio=r,max_iter=3000)
            elasticModel.fit(x,y)
            getPred=elasticModel.predict(X_Validate[:,1:]).reshape(-1,1)
        
            MSE=sum((Y_Validate-getPred)**2)
            if MSE< bestMSE:
                bestMSE=MSE
                bestAlpha=l1
                bestRatio=r
                bestElasticInt=elasticModel.intercept_
                bestElasticCoef=elasticModel.coef_
                bestElasticWeights=np.concatenate((bestElasticInt,bestElasticCoef)).reshape(-1,1)
                cacheWeights['Elastic']=bestElasticWeights


    return(bestElasticWeights)


# Get Lasso Weights
Wlasso=ChooseLambdaLasso(X_train[:,1:],Y_train)

# Get Elastic Net Weights
Welastic=ChooseParametersElasticNet(X_train[:,1:],Y_train)

#Calculate test MSE for Lasso and Elastic Net

LassoMSE=MSE(x_test_scaled,y_test,Wlasso)
ElasticMSE=MSE(x_test_scaled,y_test,Welastic)
