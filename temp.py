import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial.distance import pdist, squareform
from numpy import linalg as LA



df = pd.read_excel("HW_TESLA.xlt")
features = df.columns

# df = df.sample(frac=1).reset_index(drop=True)

data = df.loc[:, features[1:]].values
target = df.loc[:, ['STATIC']].values

x = df.loc[:, features[1:]].values
s = 0.5
pairwise_dists = squareform(pdist(x, 'euclidean'))
K = scipy.exp(-pairwise_dists ** 2 / s ** 2)
row_sum_of_K = K.sum(axis=0)
D = np.diag(row_sum_of_K)
D = np.linalg.inv(D)

DiffusionMatix = D.dot(K)

e_vals, e_vecs = LA.eig(DiffusionMatix)

print(e_vals)
print(e_vecs.shape)

mapped_data = ((e_vecs[:, 0:3]).transpose()).dot(x)

print(mapped_data)
print(mapped_data.shape)

print(e_vecs.shape)
# print(DMat.shape)
# print(K.shape)
