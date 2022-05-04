import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

classes = ['malignant', 'benign']

clf1 = svm.SVC(kernel="linear", C=1)

clf2 = KNeighborsClassifier(n_neighbors=7)

clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)

y_pred1 = clf1.predict(x_test)
y_pred2 = clf2.predict(x_test)

acc1 = metrics.accuracy_score(y_test, y_pred1)
acc2 = metrics.accuracy_score(y_test, y_pred2)
print(acc1)
print(acc2)
