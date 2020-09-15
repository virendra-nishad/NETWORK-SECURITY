import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from os.path import dirname, join
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.preprocessing import StandardScaler

# feature = []
# for i in range(42):
#     feature.append('col_' + str(i))


current_dir = dirname(__file__)
file_path = join(current_dir, 'small.txt')
dataset = pd.read_csv(file_path, sep=",", header=None)

# print(dataset.shape)

dataset = pd.DataFrame.drop_duplicates(dataset)

#print(dataset.shape)

# print(dataset.dtypes)

# obj_df = dataset.select_dtypes(include=['object']).copy()

# print(obj_df['col_1'].value_counts())
# print(obj_df['col_2'].value_counts())
# print(obj_df['col_3'].value_counts())
# print(obj_df['col_41'].value_counts())


# dataset_attack = dataset[dataset.iloc[:, 41] != 'normal.']

dos = dict.fromkeys(["back.", "land.", "neptune.", "pod.",
                     "teardrop.", "smurf."], "dos.")
r2l = dict.fromkeys(["ftp_write.", "guess_passwd.",
                     "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."], "r2l.")
u2r = dict.fromkeys(["buffer_overflow.", "loadmodule.",
                     "perl.", "rootkit."], "u2r.")
probe = dict.fromkeys(["ipsweep.", "nmap.", "portsweep.", "satan."], "probe.")

col41 = {**dos, **r2l, **u2r, **probe}

dataset = dataset.replace(col41)

labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
labelencoder_X_3 = LabelEncoder()
labelencoder_X_1.fit(dataset.iloc[:, 1])
labelencoder_X_2.fit(dataset.iloc[:, 2])
labelencoder_X_3.fit(dataset.iloc[:, 3])
dataset.iloc[:, 1] = labelencoder_X_1.transform(dataset.iloc[:, 1])
dataset.iloc[:, 2] = labelencoder_X_2.transform(dataset.iloc[:, 2])
dataset.iloc[:, 3] = labelencoder_X_3.transform(dataset.iloc[:, 3])

# print((dataset.iloc[:, 3]).unique())

#dataset['col_1'] = dataset['col_1'].astype('category')
# dataset['col_2'] = dataset['col_2'].astype('category')
# dataset['col_3'] = dataset['col_3'].astype('category')
# dataset['col_41'] = dataset['col_41'].astype('category')
# obj_df = obj_df.replace(col41)

# print(obj_df['col_41'].value_counts())

# obj_df['col_1'] = obj_df['col_1'].astype('category')

# obj_df["col_1_cat"] = obj_df["col_1"].cat.codes
# print(obj_df.head())
# print(obj_df['col_1'].value_counts())

# print(dataset.dtypes)

#col1 = dict.fromkeys({'tcp':1, "udp":2, "icmp":3})
#dataset = dataset.replace(col1)

#print((dataset.iloc[:, 1]).unique())

array = dataset.values
X = dataset.iloc[:, 0:41]
Y = dataset.iloc[:, 41]

# validation_size = 0.25
# seed = 7
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
#     X, Y, test_size=validation_size, random_state=seed)

# clf = SVC(gamma='auto')

# start_time = time.time()
# clf.fit(X_train, Y_train)
# end_time = time.time()
# print("Time taken for data training : ", end_time - start_time)

# start_time = time.time()
# predictions = clf.predict(X_validation)
# end_time = time.time()
# print("Time taken for prediction : ",  end_time - start_time)

# print("Prediction accuracy: ", accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

array = dataset.values
X = dataset.iloc[:, 0:41]
Y = dataset.iloc[:, 41]

validation_size = 0.25
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=validation_size, random_state=seed)

# scoring = 'accuracy'
clf = DecisionTreeClassifier()

start_time = time.time()
clf.fit(X_train, Y_train)
end_time = time.time()
print("Time taken for data training : ", end_time - start_time)

start_time = time.time()
predictions = clf.predict(X_validation)
end_time = time.time()
print("Time taken for prediction : ",  end_time - start_time)

print("Prediction accuracy: ", accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
