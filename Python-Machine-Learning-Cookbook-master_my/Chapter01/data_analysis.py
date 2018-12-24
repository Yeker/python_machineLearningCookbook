# coding:utf-8
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd

# 主成分分析
from sklearn import datasets,decomposition,manifold

if __name__=='__main__':
    # Load housing data
    housing_data = datasets.load_boston()
    # Shuffle the data
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)
    # print(housing_data.feature_names)
    df = pd.DataFrame(X)
    df.columns = housing_data.feature_names
    df['target'] = y

    # df.to_excel('housing.xlsx',index=False, index_label=None)
    # 朱成分分析
    pca = decomposition.PCA(n_components=None)
    pca.fit(X)
    X_r=pca.transform(X)
    print("各主成分的方差值:"+str(pca.explained_variance_))
    print("各主成分的方差值比:"+str(pca.explained_variance_ratio_))