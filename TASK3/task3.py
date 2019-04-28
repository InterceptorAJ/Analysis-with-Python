import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import learning_curve


X = []
y = []
results = []
title = "Wykres z przedstawioną skutecznością klasyfikatora"
sonar_data = pd.read_csv(f'sonar.csv')
sonar_data = pd.read_csv(f'sonar.csv',
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
            "S", "T", "U", "V", "W", "X", "Y", "Z", "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
            "I1", "J1", "K1", "L1", "M1", "N1", "O1", "P1", "Q1", "R1", "S1", "T1", "U1", "V1", "W1",
            "X1", "Y1", "Z1", "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "Class"])

X = sonar_data.drop("Class", axis=1)
y = sonar_data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

# SVC linear classifier
def svc_linear():
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją liniową')
    global a2
    a2 = SVC(kernel='linear', gamma='scale', max_iter=-1)
    a2.fit(X_train, y_train)
    y_pred = a2.predict(X_test)
    labels = ['M', 'R']
    cm = confusion_matrix(y_test,y_pred, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Macierz klasyfikacji dla funkcji liniowej')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywitość')
    global a1
    a1 = plt
    print(classification_report(y_test,y_pred))
    global a
    a = (a2.score(X,y))
    a = round(a, 2)
    results.append(a)
    print(f'Wynik uzyskany')
    print(a)


#SVC polymnial classifier
def svc_polymnial():
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją wielomianową')
    global b2
    b2 = SVC(kernel='poly', gamma="scale", max_iter=-1)
    b2.fit(X_train, y_train)
    y_pred = b2.predict(X_test)
    labels = ['M', 'R']
    cm = confusion_matrix(y_test,y_pred, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Macierz klasyfikacji dla funkcji wielomianowej')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywitość')
    global b1
    b1 = plt
    print(classification_report(y_test, y_pred))
    global b
    b = (b2.score(X,y))
    b = round(b, 2)
    results.append(b)
    print(f'Wynik uzyskany')
    print(b)


# SVC RBF classifier
def svc_rbf():
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją gaussowską')
    global c2
    c2 = SVC(kernel='rbf', gamma="scale", max_iter=-1)
    c2.fit(X_train, y_train)
    y_pred = c2.predict(X_test)
    labels = ['M', 'R']
    cm = confusion_matrix(y_test,y_pred, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Macierz klasyfikacji dla funkcji gaussowskiej')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywitość')
    global c1
    c1 = plt
    print(classification_report(y_test, y_pred))
    global c
    c = (c2.score(X,y))
    c = round(c, 2)
    results.append(c)
    print(f'Wynik uzyskany')
    print(c)


# SVC sigmoid classifier
def svc_sigmoid():
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją sigmoidalną')
    global d2
    d2 = SVC(kernel='sigmoid', gamma="scale", max_iter=-1)
    d2.fit(X_train, y_train)
    y_pred = d2.predict(X_test)
    labels = ['M', 'R']
    cm = confusion_matrix(y_test,y_pred, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Macierz klasyfikacji dla funkcji sigmoidalnej')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predykcja')
    plt.ylabel('Rzeczywitość')
    global d1
    d1 = plt
    print(classification_report(y_test, y_pred))
    global d
    d = (d2.score(X,y))
    d = round(d, 2)
    results.append(d)
    print(f'Wynik uzyskany')
    print(d)

svc_linear()
svc_polymnial()
svc_rbf()
svc_sigmoid()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Wynik")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(b2, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)


if max(results) == a:
    print(f'Najbardziej dokładny jest klasyfikator oparty o funkcje liniową')
    print(max(results))
    a1.show()
    plot_learning_curve(a2, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

if max(results) == b:
    print(f'Najbardziej dokładny jest klasyfikator oparty o funkcje wielomianową')
    print(max(results))
    b1.show()
    plot_learning_curve(b2, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

if max(results) == c:
    print(f'Najbardziej dokładny jest klasyfikator oparty o funkcje gaussowską')
    print(max(results))
    c1.show()
    plot_learning_curve(c2, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

if max(results) == d:
    print(f'Najbardziej dokładny jest klasyfikator oparty o funkcje sigmoidalną')
    print(max(results))
    d1.show()
    plot_learning_curve(d2, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
