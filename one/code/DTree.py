import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.metrics import confusion_matrix
import pydotplus
import time

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

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

start1 = time.time()
clf = clf.fit(train_data, train_lbl)
end1 = time.time()

start2 = time.time()
y_pred = clf.predict(test_data)
end2 = time.time()

print("Time taken to build decision tree: ", end1 - start1)
print("Time taken for prediction: ", end2 - start2)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(test_lbl, y_pred))
tn, fp, fn, tp = confusion_matrix(test_lbl, y_pred).ravel()
print("True Negative : ", tn)
print("True Positive : ", tp)
print("False Negative : ", fn)
print("False Positive : ", fp)


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True,
                special_characters=True, feature_names=features[1:],
                class_names=['0', '1'], label='all', node_ids=True, proportion=True,
                impurity=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Grid.png')
Image(graph.create_png())