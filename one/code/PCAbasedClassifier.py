import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import time
import numpy as np

df = pd.read_excel("HW_TESLA.xlt")
features = df.columns
# shuffling data
df = df.sample(frac=1).reset_index(drop=True)
# apply normalisation on features only not label which is column 0
data = df.loc[:, features[1:]].values
target = df.loc[:, ['STATIC']].values

# test_size: what proportion of original data is used for test set
temp_data, validate_data, temp_lbl, validate_lbl = train_test_split(data, target,
                                                             test_size=1/4.0, random_state=0)
                                                             
train_data, test_data, train_lbl, test_lbl = train_test_split(temp_data, temp_lbl,
                                                              test_size=1/3.0, random_state=0)

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_data)

# Apply transform to both the training set and the test set.
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# Make an instance of the Model
pca = PCA(n_components=3)

pca.fit(train_data)

train_data = pca.transform(train_data)
test_data = pca.transform(test_data)

clf = SVC(kernel='linear')
start1 = time.time()
clf.fit(train_data, train_lbl)
end1 = time.time()

start2 = time.time()
y_pred = clf.predict(test_data)
end2 = time.time()

print("Time taken to train data: ", end1 - start1)
print("Time taken for prediction: ", end2 - start2)

print("Accuracy:", metrics.accuracy_score(test_lbl, y_pred))
tn, fp, fn, tp = confusion_matrix(test_lbl, y_pred).ravel()

print("True Negative : ", tn)
print("True Positive : ", tp)
print("False Negative : ", fn)
print("False Positive : ", fp)

