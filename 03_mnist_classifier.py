from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve

def plotNumber(X):
    digit = X[36000];
    digit_image = digit.reshape(28, 28);
    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])

if __name__ == "__main__":
    mnist = fetch_mldata("MNIST original")
    X, y = mnist["data"], mnist["target"]
    print (X.shape, y.shape)
    #plotNumber(X)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    shuffle_index = np.random.permutation(60000);
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    y_train_5 = (y_train == 5)
    X_train_5 = (X_train == 5)

    sgd_classifier = SGDClassifier(random_state=42)
    sgd_classifier.fit(X_train, y_train_5)
    print (sgd_classifier.predict( [ X[36000]] ))

    val_score = cross_val_score(sgd_classifier, X_train, y_train_5,cv=3, scoring="accuracy")
    print (val_score)

    y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)
    print( y_train_pred )
    conf_matrix = confusion_matrix(y_train_5, y_train_pred)
    print( conf_matrix )
    print (precision_score(y_train_5, y_train_pred))
    print (recall_score(y_train_5, y_train_pred))
    print (f1_score(y_train_5, y_train_pred))

    print(sgd_classifier.decision_function([X[36000]]))
    y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3,
                                 method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()