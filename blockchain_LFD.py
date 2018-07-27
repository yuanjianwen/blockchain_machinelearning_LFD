import pandas as pd
import numpy as np
from numpy import *
import math
blockchain_data = pd.read_csv('blockchain_data.csv')
x = blockchain_data[['Difficulty','Hashrate [H/s]']].values

for i in x:
    i[0] = i[0]/(10**len(str(i[1])))
    i[1] = i[1]/(10**len(str(i[1])))
y = blockchain_data['label'].values
w = np.array([0,0])
yita = 1
x = mat(x)
w = mat(w)
############################################################
def sita(x):
    result = 1/(1+math.exp(-x))
    return result

timestamp=0
last_Ein = 100
while timestamp<10000:
    timestamp+=1
    wx = pd.DataFrame(x*w.T)#OK
    wx = list(wx.values.T[0])
    y = list(y)
    hx = []
    yx = []
    tmp_sum = 0
    gradient_tmp = 0
    for i in range(0,len(wx)):
        hx.append(sita(wx[i]))#OK
        yx.append(x[i]*y[i])#OK
        tmp_sum+= math.log(1+math.exp(-y[i]*wx[i]))
        #ywx = (1/(1+math.exp(-y[i]*wx[i])))*(-yx[i])
        #gradient_tmp += ywx
        gradient_tmp += sita(-y[i]*wx[i])*(-yx[i])
        
    Ein = tmp_sum/len(y)#这里是Ein
    if Ein > last_Ein:
        break
    last_Ein = Ein
    gradient = gradient_tmp/len(wx)#这里算出来了梯度
    
    ##############################################
    
    ##############################################
    
    #开始计算梯度的模长
    yita_tmp = 0
    for i in gradient.tolist()[0]:
        yita_tmp+=i**2
    yita_tmp = yita_tmp ** 0.5    #梯度的模长
    yita = 1/yita_tmp #固定学习率为1
    w = w - yita*gradient
    
    print(Ein,yita_tmp,w)#结果输出 Ein 梯度 w