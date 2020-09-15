import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("HW_TESLA.xlt")
features = df.columns

data = df.loc[:, features[1:]].values
target = df.loc[:, ['STATIC']].values

# test_size: what proportion of original data is used for test set
temp_data, validate_data, temp_lbl, validate_lbl = train_test_split(data, target,
                                                                    test_size=1/4.0, random_state=0)

train_data, test_data, train_lbl, test_lbl = train_test_split(temp_data, temp_lbl,
                                                              test_size=1/3.0, random_state=0)
scaler = StandardScaler()

scaler.fit(train_data)
train_data = scaler.transform(train_data)

pca = PCA(n_components=15)

pca.fit(train_data)

train_data = pca.transform(train_data)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(15))
plt.axvline(x=3, linestyle='--', c='r')
plt.xlabel('Number of Components')
plt.ylabel('Variance fraction')
plt.title('Scree plot')
plt.legend(['Cumulative variance'], prop={'size': 10})
plt.show()
