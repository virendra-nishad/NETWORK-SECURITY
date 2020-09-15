import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("HW_TESLA.xlt")

# shuffling data frame
df = df.sample(frac=1).reset_index(drop=True)

features = df.columns

# apply normalisation on features only not label which is column 0
x = df.loc[:, features[1:]].values
x = StandardScaler().fit_transform(x)

normalised_data = pd.DataFrame(x, columns=features[1:])


pca_data = PCA(n_components=2)
principalComponents_data = pca_data.fit_transform(x)

principal_data_Df = pd.DataFrame(data = principalComponents_data, columns = ['principal component 1', 'principal component 2'])

print('Fraction of data captured by two major principle component: {}'.format(pca_data.explained_variance_ratio_))
print("Total data captured by two leading principle component: ", np.sum(pca_data.explained_variance_))

plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1', fontsize=20)
plt.ylabel('Principal Component - 2', fontsize=20)
plt.title("Plot of given data set", fontsize=20)
targets = [0, 1]
colors = ['g', 'r']
for target, color in zip(targets, colors):
    indicesToKeep = df['STATIC'] == target
    plt.scatter(principal_data_Df.loc[indicesToKeep, 'principal component 1']
               , principal_data_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 20)

plt.legend(['Normal', 'Stress'], prop={'size': 15})
plt.show()