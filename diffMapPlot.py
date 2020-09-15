import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from pydiffmap import diffusion_map as dm

df = pd.read_excel("HW_TESLA.xlt")
features = df.columns

df = df.sample(frac=1).reset_index(drop=True)

data = df.loc[:, features[1:]].values
target = df.loc[:, ['STATIC']].values

x = df.loc[:, features[1:]].values
#x = StandardScaler().fit_transform(x)

normalised_data = pd.DataFrame(x, columns=features[1:])

mydmap = dm.DiffusionMap.from_sklearn(
    n_evecs=3, k=70, epsilon=10, alpha=0.1)

stressT = mydmap.fit_transform(normalised_data)

principal_data_Df = pd.DataFrame(data=stressT, columns=[
                                 'principal component 1',
                                 'principal component 2',
                                 'principal component 3'])


fig = plt.figure()
ax = plt.axes(projection='3d')

targets = [0, 1]
colors = ['g', 'r']
for target, color in zip(targets, colors):
    indicesToKeep = df['STATIC'] == target
    ax.scatter3D(principal_data_Df.loc[indicesToKeep, 'principal component 1'],
                  principal_data_Df.loc[indicesToKeep, 'principal component 2'],
                  principal_data_Df.loc[indicesToKeep, 'principal component 3'],
                  c=color, s=40)

plt.legend(['Normal', 'Stress'], prop={'size': 15})
plt.show()
