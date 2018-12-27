# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

# 局部加权
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


####分析预测误差的大小
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


########岭回归#########
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #to eliminate X0 take mean off of Y
    #regularize X's
    xMeans = mean(xMat,0)   #calc mean then subtract it off
    xVar = var(xMat,0)      #calc variance of Xi then divide by it
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        wMat[i,:]=ws.T
    return wMat



if __name__=='__main__':
    xArr,yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr,yArr)
    # print(xArr[0:2])
    # print(yArr[0])
    # xMat = mat(xArr)
    # yMat = mat(yArr)
    # yHat = xMat*ws
    # fig = plt.figure()
    # temp1 = xMat[:,1].flatten().A[0]
    # temp2 = yMat.T[:,0].flatten().A[0]
    # ax = fig.add_subplot(111)
    # ax.scatter(temp1,temp2)
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy*ws
    # ax.plot(xCopy[:,1],yHat)
    # plt.show()
    ########### 局部加权########
    # yHat = lwlrTest(xArr,xArr,yArr,0.01)
    # xMat = mat(xArr)
    # srtInd = xMat[:,1].argsort(0)
    # xSort = xMat[srtInd][:,0,:]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(xSort[:,1],yHat[srtInd])
    # temp1 = xMat[:,1].flatten().A[0]
    # temp2 = mat(yArr).T.flatten().A[0]
    # ax.scatter(temp1,temp2,s = 2,c = 'red')
    # plt.show()
    # ###########预测鲍鱼的年龄##########
    # abX,abY = loadDataSet('abalone.txt')
    # yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    # yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    # yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    # print(rssError(abY[0:99],yHat01.T))
    # print(rssError(abY[0:99],yHat1.T))
    # print(rssError(abY[0:99],yHat10.T))
    # #新数据上的表现
    # yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    # yHat1 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    # yHat10 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
    # print(rssError(abY[100:199],yHat01.T))
    # print(rssError(abY[100:199],yHat1.T))
    # print(rssError(abY[100:199],yHat10.T))
    # #与简单的线性回归做比较
    # ws = standRegres(abX[0:99],abY[0:99])
    # yHat = mat(abX[100:199])*ws
    # print(rssError(abY[100:199],yHat.T.A))
    ##########岭回归#################
