# coding:utf-8

#chap17 regression
import pandas as pd
TRD_Index=pd.read_table('017/TRD_Index.txt',sep='\t')
# 上证综合指数
SHindex=TRD_Index[TRD_Index.Indexcd==1]
# 深证综指收益率
SZindex=TRD_Index[TRD_Index.Indexcd==399106]
SHRet=SHindex.Retindex
SZRet=SZindex.Retindex
SZRet.index=SHRet.index


import statsmodels.api as sm
model=sm.OLS(SHRet,sm.add_constant(SZRet)).fit()
print(model.summary())
print(model.fittedvalues[:5])

import matplotlib.pyplot as plt
plt.scatter(model.fittedvalues,model.resid)
plt.xlabel('拟合值')
plt.ylabel('残差')

import scipy.stats as stats
sm.qqplot(model.resid_pearson,\
              stats.norm,line='45')

plt.scatter(model.fittedvalues,\
             model.resid_pearson**0.5)
plt.xlabel('拟合值')
plt.ylabel('标准化残差的平方根')

penn=pd.read_excel('017/Penn World Table.xlsx',2)
penn.head(3)
import numpy as np
model=sm.OLS(np.log(penn.rgdpe),
             sm.add_constant(penn.iloc[:,-6:])).fit()
print(model.summary())

penn.iloc[:,-6:].corr()

model=sm.OLS(np.log(penn.rgdpe),\
             sm.add_constant(penn.iloc[:,-5:-1])).fit()
print(model.summary())
plt.show()