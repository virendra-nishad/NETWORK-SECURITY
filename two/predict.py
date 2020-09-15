import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pydiffmap import diffusion_map as dm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import metrics

df = pd.read_excel("HW_TESLA.xlt")
features = df.columns
df = df.sample(frac=1).reset_index(drop=True)
data = df.loc[:, features[1:]].values
target = df.loc[:, ['STATIC']].values

train_data, test_data, train_lbl, test_lbl = train_test_split(data, target,
                                                              test_size=1/3.0, random_state=0)

mydmap = dm.DiffusionMap.from_sklearn(
    n_evecs=3, epsilon=10, alpha=0.1, k=70)

mydmap.fit(train_data)

train_data = mydmap.transform(train_data)
test_data = mydmap.transform(test_data)


clf = KMeans(n_clusters=2, random_state=0)
clf.fit(train_data, train_lbl)

y_pred = clf.predict(test_data)

print("Accuracy:", metrics.accuracy_score(test_lbl, y_pred))
tn, fp, fn, tp = confusion_matrix(test_lbl, y_pred).ravel()

print("True Negative : ", tn)
print("True Positive : ", tp)
print("False Negative : ", fn)
print("False Positive : ", fp)
