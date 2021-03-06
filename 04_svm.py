import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model LogisticRegression


def test_iris():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)] # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])
    svm_clf.fit(X, y)


