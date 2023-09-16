#HW3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def evaluate( PrtrainY, trainY, PrtestY, testY): 
    rmse_tr=np.sqrt(mean_squared_error(trainY, PrtrainY))
    rmse_te=np.sqrt(mean_squared_error(testY, PrtestY))
    mae_tr=mean_absolute_error(trainY, PrtrainY)
    mae_te=mean_absolute_error(testY, PrtestY)
    #print("RMSE train:", rmse_tr)
    #print("RMSE test:", rmse_te)
    #print("MAE train:", mae_tr)
    #print("MAE test:", mae_te)
    return [rmse_tr, rmse_te, mae_tr, mae_te]

def GD(N,x,y,w,alpha):
    temp=[0,0]
    s1=0
    s2=0
    for i in range(0,len(x)):
        y_hat=w[0]+(w[1]*x[i][1])
        s1+=y_hat-y[i]
        s2+=(y_hat-y[i])*x[i][1]
    temp[0]=w[0]-alpha*(float(1/(2*N))*s1)
    temp[1]=w[1]-alpha*(float(1/(2*N))*s2)
    temp=np.array(temp)
    return temp

def SGD(data2,w,alpha):
    temp=[0,0]
    s=data2.sample()
    y_hat=w[0]+(w[1]*s["x"])
    temp[0]=w[0]-alpha*(float(y_hat-float(s["t"])))
    temp[1]=w[1]-alpha*(float(y_hat-float(s["t"]))*float(s["x"]))
    temp=np.array(temp)
    return temp

#print(data2.sample())
##########
data2=pd.read_csv('data2.csv') 
X2=data2["x"]
y2=data2["t"]
data3=pd.read_csv('data3.csv') 
X3=data3["x"]
y3=data3["t"]


########## gradient decsent
np.random.seed(1)
w0=np.random.rand()
w1=np.random.rand()
w=np.array([w0,w1]) #initialize
W=w
print(w)
phiTr=[]
phiTe=[]
for i in X2:
    x=[1,i]
    phiTr.append(x)   
phiTr=np.array(phiTr)
for i in X3:
    x=[1,i]
    phiTe.append(x)   
phiTe=np.array(phiTe)
alpha=0.01
gdtr=[]
gdte=[]
for i in range(0,300):
    w=GD(len(X2),phiTr,y2,w,alpha)
    y_pred_tr=np.dot(phiTr,w.transpose())
    y_pred_te=np.dot(phiTe,w.transpose())
    e=evaluate(y_pred_tr,y2,y_pred_te,y3)
    gdtr.append(e[0])
    gdte.append(e[1])

bestRmse=min(gdte)
itr=gdte.index(bestRmse) +1
print("the best rmse for test on GD is :",bestRmse,"on the",itr,"iteration")


#plot    
iteration=[i for i in range(1,301)]
plt.figure()
plt.scatter(iteration, gdtr)
plt.ylabel("RMSE train uisng Gradient Decsent")
plt.xlabel("number of iteration")
plt.show(block=False)    

plt.figure()
plt.scatter(iteration, gdte)
plt.ylabel("RMSE test uisng Gradient Decsent")
plt.xlabel("number of iteration")
plt.show(block=False)            



###########SGD 
np.random.seed(1)   
w=W 
print("_______________")  
sgdtr=[] 
sgdte=[]     
for  i in range(0,300):       
    w=SGD(data2,w,alpha)
    y_pred_tr=np.dot(phiTr,w.transpose())
    y_pred_te=np.dot(phiTe,w.transpose())
    e=evaluate(y_pred_tr,y2,y_pred_te,y3)
    sgdtr.append(e[0])
    sgdte.append(e[1])
    
bestRmse=min(sgdte)
itr=sgdte.index(bestRmse)+1
print("the best rmse for test on GD is :",bestRmse,"on the",itr,"iteration")
   
#plot    
iteration=[i for i in range(1,301)]
plt.figure()
plt.scatter(iteration, sgdtr)
plt.ylabel("RMSE train uisng Stochastic Gradient Decsent")
plt.xlabel("number of iteration")
plt.show(block=False)    
          
plt.figure()
plt.scatter(iteration, sgdte)
plt.ylabel("RMSE test uisng Stochastic Gradient Decsent")
plt.xlabel("number of iteration")
plt.show(block=False)         
        
        
