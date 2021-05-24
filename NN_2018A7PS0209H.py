# Jhaveri Aditya Alok (2018A7PS0209H)
# Aryesh Harshal Koya (2018A4PS0637H)
# Vaishnavee Nautiyal (2018A7PS0286H)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Wrting Functions to cleanse the data as well as get the data through pandas

def oneHotY(y,nLabels):
	numberOfSamples=y.shape[0]
	encoded=np.zeros((nLabels,numberOfSamples))
	for i in range(numberOfSamples):

		classindx=int(y[i])-1
		encoded[classindx,i]=1

	return encoded

def Standardize(X):
	X = (X-np.mean(X,axis= 1,keepdims=True)) / np.std(X,axis= 1,keepdims=True)
	# print(X)
	return X


def SplitTrainTestData(data,nLabels):
	# np.random.shuffle(data)
	n=len(data)
	splitValue=int(0.7*n)
	trainData=data[0:splitValue]
	xtrain,ytrain=(trainData[:,:-1].T,trainData[:,-1].T)
	trainLabels=ytrain
	xtrain=Standardize(xtrain)
	ytrain=ytrain.reshape(len(ytrain),1)
	ytrain=oneHotY(ytrain,nLabels)


	test=data[splitValue:n]
	xtest,ytest=(test[:,:-1].T,test[:,-1].T)
	testLabels=ytest
	xtest=Standardize(xtest)
	ytest=ytest.reshape(len(ytest),1)
	ytest=oneHotY(ytest,nLabels)
	return xtrain,ytrain,xtest,ytest,testLabels,trainLabels


data=pd.read_csv('dataset_NN.csv')
data=data.to_numpy()
np.random.shuffle(data)

alpha = 0.5 # 0.01 0.5 0.05
epochs = 1000

nLabels=len(np.unique(data[:,-1]))

xtrain,ytrain,xtest,ytest,testLabels,trainLabels=SplitTrainTestData(data,nLabels)
batch_ratio=0.4
batchSize=int(batch_ratio*xtrain.shape[1])



# Defining the functions to be used for implementing a Neural Network

def plotGraph(loss,title,ylabel):
	plt.title(title)
	plt.plot(loss)
	plt.xlabel('Iterations')
	plt.ylabel(ylabel)
	plt.show()


def SoftmaxFunction(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def Sigmoid(z):
	return 1/(1+np.exp(-z))

def initParams(numberx,HiddenLayerSizes,numbery):
	h1,h2=HiddenLayerSizes[:]
	W1=np.random.normal(0,1,size=(h1,numberx))
	b1=np.zeros((h1,1))
	if(h2!=0):
		W2=np.random.normal(0,1,size=(h2,h1))
		b2=np.zeros((h2,1))
		W3=np.random.normal(0,1,size=(numbery,h2))
		b3=np.zeros((numbery,1))
		return np.array([W1,b1,W2,b2,W3,b3],dtype=object)
	
	W2=np.random.normal(0,1,size=(numbery,h1))
	b2=np.zeros((numbery,1))
	return np.array([W1,b1,W2,b2],dtype=object)

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

def accuracy(y_pred,labels):
	return np.sum(y_pred==labels)/len(labels)*100

def forwardProp(X,parameters,HiddenLayerSizes):
	h1,h2=HiddenLayerSizes[:]
	if(h2==0):
		W1,b1,W2,b2=parameters
		Z1=W1@X+b1
		A1=Sigmoid(Z1)
		Z2=W2@A1+b2
		A2=SoftmaxFunction(Z2)
		nodes=np.array([Z1,A1,Z2,A2],dtype=object)
	else:
		W1,b1,W2,b2,W3,b3=parameters
		Z1=W1@X+b1
		A1=Sigmoid(Z1)
		Z2=W2@A1+b2
		A2=Sigmoid(Z2)
		Z3=W3@A2+b3
		A3=SoftmaxFunction(Z3)
		nodes=np.array([Z1,A1,Z2,A2,Z3,A3],dtype=object)

	return nodes

def gradient(nodes,parameters,X,y,HiddenLayerSizes):
	h1,h2=HiddenLayerSizes[:]
	m=X.shape[1]
	if(h2==0):
		W1,b1,W2,b2=parameters
		Z1,A1,Z2,A2=nodes
		dz2=(A2-y)
		dw2=dz2@(A1.T)/m
		db2=np.sum(dz2,axis=1,keepdims=True)/m
		da1=(W2.T)@dz2
		dz1=da1*(A1*(1-A1))
		dw1=dz1@(X.T)/m
		db1=np.sum(dz1,axis=1,keepdims=True)/m
		grads=np.array([dw1,db1,dw2,db2],dtype=object)

	else:
		W1,b1,W2,b2,W3,b3=parameters
		Z1,A1,Z2,A2,Z3,A3=nodes
		dz3=(A3-y)
		dw3=dz3@(A2.T)/m
		db3=np.sum(dz3,axis=1,keepdims=True)/m
		da2=(W3.T)@dz3
		dz2=da2*(A2*(1-A2))
		dw2=dz2@(A1.T)/m
		db2=np.sum(dz2,axis=1,keepdims=True)/m
		da1=(W2.T)@dz2
		dz1=da1*(A1*(1-A1))
		dw1=dz1@(X.T)/m
		db1=np.sum(dz1,axis=1,keepdims=True)/m
		grads=np.array([dw1,db1,dw2,db2,dw3,db3],dtype=object)
	
	return grads


def getBatch(x,y,batchSize,labels):
	X=x.T
	Y=y.T
	labels=labels.reshape(len(labels),1)
	
	shuffler=np.random.permutation(len(X))
	X=X[shuffler][0:batchSize]
	Y=Y[shuffler][0:batchSize]
	labels=trainLabels[shuffler][0:batchSize]
	return X.T,Y.T,labels

def GradDesc(X,y,alpha,epochs,HiddenLayerSizes,batchSize,labels):
	h1,h2=HiddenLayerSizes[:]
	nLabels=y.shape[0]
	losses=np.array([])
	accuracies=np.array([])
	parameters=initParams(X.shape[0],HiddenLayerSizes,nLabels)
	
	for i in tqdm(range(epochs)):
		#X,y,labels=getBatch(X,y,batchSize,labels)
		nodes=forwardProp(X,parameters,HiddenLayerSizes)
		grads=gradient(nodes,parameters,X,y,HiddenLayerSizes)
		parameters-=alpha*grads
		pred=nodes[-1]
		
		
		loss=error(pred.T,y.T)
		#print("Loss: {:.2f}".format(loss))
		losses=np.append(losses,loss)
		pred_labels=np.argmax(pred,axis=0)+1
		acc=accuracy(pred_labels,labels)
		accuracies=np.append(accuracies,acc)

	return parameters,losses,accuracies



# For one layered ANN

HiddenLayerSizes=[128,0]

print("One Layered Neural Network's results are as follow:")
parameters,lossOne,accuracyOne=GradDesc(xtrain,ytrain,alpha,epochs,HiddenLayerSizes,batchSize,trainLabels)
trainPred=np.argmax(forwardProp(xtrain,parameters,HiddenLayerSizes)[-1],axis=0)+1
trainAccuracy=accuracy(trainPred,trainLabels)
print("Training Accuracy for {:.3f}".format(trainAccuracy)+"%" + " for learning rate:", alpha)


testPred=np.argmax(forwardProp(xtest,parameters,HiddenLayerSizes)[-1],axis=0)+1
testAccuracy=accuracy(testPred,testLabels)
print("Testing Accuracy {:.3f}".format(testAccuracy)+"%"+ " for learning rate:", alpha)
print("Final Loss {:.3f}".format(lossOne[-1])+"\n")

# For 2-layered ANN

HiddenLayerSizes=[128,128]

print("Two Layered Neural Network's results are as follow:")
parameters,lossTwo,accuracyTwo=GradDesc(xtrain,ytrain,alpha,epochs,HiddenLayerSizes,batchSize,trainLabels)
trainPred=np.argmax(forwardProp(xtrain,parameters,HiddenLayerSizes)[-1],axis=0)+1
trainAccuracy=accuracy(trainPred,trainLabels)
print("Training Accuracy {:.3f}".format(trainAccuracy)+"%" + " for learning rate:", alpha)


testPred=np.argmax(forwardProp(xtest,parameters,HiddenLayerSizes)[-1],axis=0)+1
testAccuracy=accuracy(testPred,testLabels)
print("Testing Accuracy {:.3f}".format(testAccuracy)+"%" + " for learning rate:", alpha)
print("Final Loss {:.3f}".format(lossTwo[-1]))


plotGraph(lossOne,'One hidden layer','Loss')
plotGraph(accuracyOne,'One hidden layer','Accuracy')
plotGraph(lossTwo,'Two hidden layers','Loss')
plotGraph(accuracyTwo,'Two hidden layers','Accuracy')

