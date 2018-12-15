# coding:utf-8
import numpy as np
from sklearn import preprocessing
# 程序说明：
# 数据预处理演示


# 样本数据
data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])

# mean removal
# 特征值减去均值，除以标准差；会导致特征均值为0，标准差为1
data_standardized = preprocessing.scale(data)
print("\nMean =", data_standardized.mean(axis=0))
print("Std deviation =", data_standardized.std(axis=0))

# min max scaling
# 范围缩放：最小-最大规范化对原始数据进行线性变换，变换到[0,1]区间（也可以是其他固定最小最大值的区间）
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print ("\nMin max scaled data:\n", data_scaled)

# normalization
# 规范化（Normalization）
# 规范化是将不同变化范围的值映射到相同的固定范围，常见的是[0,1]，此时也称为归一化。《机器学习》周志华
data_normalized = preprocessing.normalize(data, norm='l1')
print ("\nL1 normalized data:\n", data_normalized)

# binarization
# 特征二值化（Binarization）
# 给定阈值，将特征转换为0/1
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print ("\nBinarized data:\n", data_binarized)

# one hot encoding
# 类别特征编码
# 有时候特征是类别型的，而一些算法的输入必须是数值型，此时需要对其编码。
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\nEncoded vector:\n", encoded_vector)

