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
# def ridgeRegres(xMat,yMat,lam=0.2):
#     xTx = xMat.T*xMat
#     denom = xTx + eye(shape(xMat)[1])*lam
#     if linalg.det(denom) == 0.0:
#         print("This matrix is singular, cannot do inverse")
#         return
#     ws = denom.I * (xMat.T*yMat)
#     return ws

# def ridgeTest(xArr,yArr):
#     xMat = mat(xArr); yMat=mat(yArr).T
#     yMean = mean(yMat,0)
#     yMat = yMat - yMean     #to eliminate X0 take mean off of Y
#     #regularize X's
#     xMeans = mean(xMat,0)   #calc mean then subtract it off
#     xVar = var(xMat,0)      #calc variance of Xi then divide by it
#     xMat = (xMat - xMeans)/xVar
#     numTestPts = 30
#     wMat = zeros((numTestPts,shape(xMat)[1]))
#     for i in range(numTestPts):
#         ws = ridgeRegres(xMat,yMat,exp(i-10))
#         wMat[i,:]=ws.T
#     return wMat



#########向前逐步回归#########
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

##########乐高数据采集（Google API）（已经关闭没有用了）##########
# from time import sleep
# import json
# import urllib.request
# def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
#     sleep(10)
#     myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
#     searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
#     pg = urllib.request.urlopen(searchURL)
#     retDict = json.loads(pg.read())
#     for i in range(len(retDict['items'])):
#         try:
#             currItem = retDict['items'][i]
#             if currItem['product']['condition'] == 'new':
#                 newFlag = 1
#             else: newFlag = 0
#             listOfInv = currItem['product']['inventories']
#             for item in listOfInv:
#                 sellingPrice = item['price']
#                 if  sellingPrice > origPrc * 0.5:
#                     print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
#                     retX.append([yr, numPce, newFlag, origPrc])
#                     retY.append(sellingPrice)
#         except: print('problem with item %d' % i)
#
# def setDataCollect(retX, retY):
#     searchForSet(retX, retY, 8288, 2006, 800, 49.99)
#     searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
#     searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
#     searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
#     searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
#     searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
#
# def crossValidation(xArr,yArr,numVal=10):
#     m = len(yArr)
#     indexList = range(m)
#     errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
#     for i in range(numVal):
#         trainX=[]; trainY=[]
#         testX = []; testY = []
#         random.shuffle(indexList)
#         for j in range(m):#create training set based on first 90% of values in indexList
#             if j < m*0.9:
#                 trainX.append(xArr[indexList[j]])
#                 trainY.append(yArr[indexList[j]])
#             else:
#                 testX.append(xArr[indexList[j]])
#                 testY.append(yArr[indexList[j]])
#         wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
#         for k in range(30):#loop over all of the ridge estimates
#             matTestX = mat(testX); matTrainX=mat(trainX)
#             meanTrain = mean(matTrainX,0)
#             varTrain = var(matTrainX,0)
#             matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
#             yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
#             errorMat[i,k]=rssError(yEst.T.A,array(testY))
#             #print errorMat[i,k]
#     meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
#     minMean = float(min(meanErrors))
#     bestWeights = wMat[nonzero(meanErrors==minMean)]
#     #can unregularize to get model
#     #when we regularized we wrote Xreg = (x-meanX)/var(x)
#     #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
#     xMat = mat(xArr); yMat=mat(yArr).T
#     meanX = mean(xMat,0); varX = var(xMat,0)
#     unReg = bestWeights/varX
#     print("the best model from Ridge Regression is:\n",unReg)
#     print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))
from numpy import *
from bs4 import BeautifulSoup

# 从页面读取数据，生成retX和retY列表
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):

    # 打开并读取HTML文件
    fr = open(inFile);
    soup = BeautifulSoup(fr.read())
    i=1

    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()

        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)

# 依次读取六种乐高套装的数据，并生成数据矩阵
def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'setHtml/lego10196.html', 2009, 3263, 249.99)

# 计算给定lambda值得回归系数
def ridgeRegres(xMat,yMat,lam=0.2):
    # 使用矩阵运算实现146页的回归系数计算公式
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam

    # 判断是否为奇异矩阵
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

# 计算回归系数矩阵
def ridgeTest(xArr,yArr):
    # 初始化X和Y矩阵
    xMat = mat(xArr); yMat=mat(yArr).T

    # 对X和Y矩阵进行标准化
    # 计算所有特征的均值
    yMean = mean(yMat,0)

    # 特征值减去各自的均值
    yMat = yMat - yMean

    # 标准化X矩阵数据
    # 获得均值
    xMeans = mean(xMat,0)
    # 获得方差
    xVar = var(xMat,0)
    # 标准化方法：减去均值除以方差
    xMat = (xMat - xMeans)/xVar

    # 计算回归系数30次
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

# 交叉验证测试岭回归
def crossValidation(xArr,yArr,numVal=10):
    # 获得数据点个数，xArr和yArr具有相同长度
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))

    # 主循环 交叉验证循环
    for i in range(numVal):
        # 随机拆分数据，将数据分为训练集（90%）和测试集（10%）
        trainX=[]; trainY=[]
        testX = []; testY = []

        # 对数据进行混洗操作
        random.shuffle(indexList)

        # 切分训练集和测试集
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])

        # 获得回归系数矩阵
        wMat = ridgeTest(trainX,trainY)

        # 循环遍历矩阵中的30组回归系数
        for k in range(30):
            # 读取训练集和数据集
            matTestX = mat(testX); matTrainX=mat(trainX)
            # 对数据进行标准化
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain

            # 测试回归效果并存储
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)

            # 计算误差
            errorMat[i,k] = ((yEst.T.A-array(testY))**2).sum()

    # 计算误差估计值的均值
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]

    # 不要使用标准化的数据，需要对数据进行还原来得到输出结果
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX

    # 输出构建的模型
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))










if __name__=='__main__':
    # xArr,yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr,yArr)
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
    # # ###########预测鲍鱼的年龄##########
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
    # abX,abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX,abY)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()
    ##########向前逐步回归###########
    # xArr,yArr = loadDataSet('abalone.txt')
    # returnMat = stageWise(xArr,yArr,0.001,5000)
    # print(returnMat)
    ##########预测乐高玩具套装价格#############
    lgX = []
    lgY = []

    setDataCollect(lgX, lgY)
    crossValidation(lgX, lgY, 10)