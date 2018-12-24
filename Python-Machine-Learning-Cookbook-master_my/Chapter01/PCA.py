import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold
from itertools import cycle
def load_data():
    iris=datasets.load_iris()
    return iris.data,iris.target


PCA_Set=[
    decomposition.PCA(n_components=None),
    decomposition.PCA(svd_solver = 'randomized'),
    decomposition.SparsePCA(n_components=None),
    decomposition.IncrementalPCA(n_components=None),
    decomposition.KernelPCA(n_components=None,kernel='linear'),
    decomposition.KernelPCA(n_components=None,kernel='rbf'),
    decomposition.KernelPCA(n_components=None,kernel='poly'),
    decomposition.KernelPCA(n_components=None,kernel='sigmoid'),
    decomposition.FastICA(n_components=None)
    ]
PCA_Set_Name=[
    'Default',
    'Randomized',
    'Sparse',
    'Incremental',
    'Kernel(linear)',
    'Kernel(rbf)',
    'Kernel(poly)',
    'Kernel(sigmoid)',
    'ICA'
    ]



def plot_PCA(*data):
    X,Y=data
    fig=plt.figure("PCA",figsize=(20, 8))

    ax=fig.add_subplot(2,5,1)
    colors=cycle('rgbcmykw')
    for label,color in zip(np.unique(Y),colors):
        position=Y==label
        ax.scatter(X[position,0],X[position,1],label="target=%d"%label,color=color)
    plt.xticks(fontsize=10, color="darkorange")
    plt.yticks(fontsize=10, color="darkorange")
    ax.set_title('Original')

    for i,PCA in enumerate(PCA_Set):
        pca=PCA
        pca.fit(X)
        X_r=pca.transform(X)

        if i==0:
            print("各主成分的方差值:"+str(pca.explained_variance_))
            print("各主成分的方差值比:"+str(pca.explained_variance_ratio_))

        ax=fig.add_subplot(2,5,i+2)
        colors=cycle('rgbcmykw')
        for label,color in zip(np.unique(Y),colors):
            position=Y==label
            ax.scatter(X_r[position,0],X_r[position,1],label="target=%d"%label,color=color)
        plt.xticks(fontsize=10, color="darkorange")
        plt.yticks(fontsize=10, color="darkorange")
        ax.set_title(PCA_Set_Name[i])
    plt.show()

X,Y=load_data()
# print(X)
# print(Y)
plot_PCA(X,Y)
