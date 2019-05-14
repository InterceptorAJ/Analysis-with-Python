from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import unique_labels



results1 = []
results2 = []
title = "Wykres z przedstawioną skutecznością klasyfikatora"
sonar_data = pd.read_csv(f'sonar.csv',
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
            "S", "T", "U", "V", "W", "X", "Y", "Z", "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
            "I1", "J1", "K1", "L1", "M1", "N1", "O1", "P1", "Q1", "R1", "S1", "T1", "U1", "V1", "W1",
            "X1", "Y1", "Z1", "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "Class"])
X = sonar_data.drop("Class", axis=1)
y = sonar_data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
cv = ShuffleSplit(test_size=0.2, random_state=0)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['Mina', 'Skała']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Rzeczywistość',
           xlabel='Predykcja')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# SVC linear classifier
def svc_linear(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją liniową')
    global a2
    a2 = SVC(kernel='linear', gamma='scale', max_iter=-1)
    a2.fit(X_train, y_train)
    global y_pred
    y_pred = a2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji liniowej')
    plt.savefig(f"{pierwszy}liniowy.png")
    print(classification_report(y_test,y_pred))
    global a
    a = (a2.score(X_test,y_test))
    a = round(a, 2)
    results.append(a)
    print(f'Wynik uzyskany')
    print(a)
    return a


#SVC polymnial classifier
def svc_polymnial(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją wielomianową')
    global b2
    b2 = SVC(kernel='poly', gamma="scale", max_iter=-1)
    b2.fit(X_train, y_train)
    y_pred = b2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji wielomianowej')
    plt.savefig(f"{pierwszy}wielomianowy.png")
    print(classification_report(y_test, y_pred))
    global b
    b = (b2.score(X_test,y_test))
    b = round(b, 2)
    results.append(b)
    print(f'Wynik uzyskany')
    print(b)
    return b


# SVC RBF classifier
def svc_rbf(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją gaussowską')
    global c2
    c2 = SVC(kernel='rbf', gamma="scale", max_iter=-1)
    c2.fit(X_train, y_train)
    y_pred = c2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji gaussowskiej')
    plt.savefig(f"{pierwszy}gauss.png")
    print(classification_report(y_test, y_pred))
    global c
    c = (c2.score(X_test,y_test))
    c = round(c, 2)
    results.append(c)
    print(f'Wynik uzyskany')
    print(c)
    return c


# SVC sigmoid classifier
def svc_sigmoid(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją sigmoidalną')
    global d2
    d2 = SVC(kernel='sigmoid', gamma="scale", max_iter=-1)
    d2.fit(X_train, y_train)
    y_pred = d2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji sigmoidalnej')
    plt.savefig(f"{pierwszy}sigmoid.png")
    print(classification_report(y_test, y_pred))
    global d
    d = (d2.score(X_test,y_test))
    d = round(d, 2)
    results.append(d)
    print(f'Wynik uzyskany')
    print(d)
    return d


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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import unique_labels



results1 = []
results2 = []
title = "Wykres z przedstawioną skutecznością klasyfikatora"
sonar_data = pd.read_csv(f'sonar.csv',
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
            "S", "T", "U", "V", "W", "X", "Y", "Z", "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1",
            "I1", "J1", "K1", "L1", "M1", "N1", "O1", "P1", "Q1", "R1", "S1", "T1", "U1", "V1", "W1",
            "X1", "Y1", "Z1", "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "Class"])
X = sonar_data.drop("Class", axis=1)
y = sonar_data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
cv = ShuffleSplit(test_size=0.2, random_state=0)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ['Mina', 'Skała']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Rzeczywistość',
           xlabel='Predykcja')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# SVC linear classifier
def svc_linear(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją liniową')
    global a2
    a2 = SVC(kernel='linear', gamma='scale', max_iter=-1)
    a2.fit(X_train, y_train)
    global y_pred
    y_pred = a2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji liniowej')
    plt.savefig(f"{pierwszy}liniowy.png")
    print(classification_report(y_test,y_pred))
    global a
    a = (a2.score(X_test,y_test))
    a = round(a, 2)
    results.append(a)
    print(f'Wynik uzyskany')
    print(a)
    return a


#SVC polymnial classifier
def svc_polymnial(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją wielomianową')
    global b2
    b2 = SVC(kernel='poly', gamma="scale", max_iter=-1)
    b2.fit(X_train, y_train)
    y_pred = b2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji wielomianowej')
    plt.savefig(f"{pierwszy}wielomianowy.png")
    print(classification_report(y_test, y_pred))
    global b
    b = (b2.score(X_test,y_test))
    b = round(b, 2)
    results.append(b)
    print(f'Wynik uzyskany')
    print(b)
    return b


# SVC RBF classifier
def svc_rbf(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją gaussowską')
    global c2
    c2 = SVC(kernel='rbf', gamma="scale", max_iter=-1)
    c2.fit(X_train, y_train)
    y_pred = c2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji gaussowskiej')
    plt.savefig(f"{pierwszy}gauss.png")
    print(classification_report(y_test, y_pred))
    global c
    c = (c2.score(X_test,y_test))
    c = round(c, 2)
    results.append(c)
    print(f'Wynik uzyskany')
    print(c)
    return c


# SVC sigmoid classifier
def svc_sigmoid(X_train, X_test, y_train, y_test, results, pierwszy):
    print(f'----------------------------------------------------------------------------------------------------------')
    print(f'Klasyfikator z funkcją sigmoidalną')
    global d2
    d2 = SVC(kernel='sigmoid', gamma="scale", max_iter=-1)
    d2.fit(X_train, y_train)
    y_pred = d2.predict(X_test)
    labels = ['Mina', 'Skała']
    plot_confusion_matrix(y_test, y_pred, classes=labels, title='Macierz klasyfikacji dla funkcji sigmoidalnej')
    plt.savefig(f"{pierwszy}sigmoid.png")
    print(classification_report(y_test, y_pred))
    global d
    d = (d2.score(X_test,y_test))
    d = round(d, 2)
    results.append(d)
    print(f'Wynik uzyskany')
    print(d)
    return d


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
    plt.xlabel("Iteracja")
    plt.ylabel("Wynik")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.01,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.01, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Wynik treningowy:")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Wynik kroswalidacji")

    plt.legend(loc="best")
    return plt

pierwszy = "pierwszy"
drugi = "drugi"

def plots(pierwszy,X,y):
    title = "Wykres z przedstawioną skutecznością dla klasyfikatora liniowego"
    a3 = plot_learning_curve(a2, title, X ,y, (0.0, 1.01), cv=cv, n_jobs=4)
    a3.savefig(f"{pierwszy}liniowy-przebieg.png")
    title = "Wykres z przedstawioną skutecznością dla klasyfikatora wielomianowego"
    b3 = plot_learning_curve(b2, title, X,y, (0.0, 1.01), cv=cv, n_jobs=4)
    b3.savefig(f"{pierwszy}wielomianowy-przebieg.png")
    title = "Wykres z przedstawioną skutecznością dla klasyfikatora gaussowskiego"
    c3 = plot_learning_curve(c2, title, X,y, (0.0, 1.01), cv=cv, n_jobs=4)
    c3.savefig(f"{pierwszy}gauss-przebieg.png")
    title = "Wykres z przedstawioną skutecznością dla klasyfikatora sigmoidalnego"
    d3 = plot_learning_curve(d2, title, X,y, (0.0, 1.01), cv=cv, n_jobs=4)
    d3.savefig(f"{pierwszy}sigmoid-przebieg.png")

svc_linear(X_train, X_test, y_train, y_test, results1, pierwszy)
svc_polymnial(X_train, X_test, y_train, y_test, results1, pierwszy)
svc_rbf(X_train, X_test, y_train, y_test, results1, pierwszy)
svc_sigmoid(X_train, X_test, y_train, y_test, results1, pierwszy)
plots(pierwszy,X,y)


plt.clf()
minmax = MinMaxScaler().fit_transform(X)
pca = PCA().fit(minmax)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig("Wykres-1.png")
print("Wynik analizy dla wszystkich 60-ciu komponentów:")
print(pca.explained_variance_ratio_)


plt.clf()
pca = PCA(n_components=2).fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig("Wykres-2.png")
print("Wynik analizy głównych składowych PCA dla dwóch komponentów:")
print(pca.explained_variance_ratio_)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

svc_linear(X_train, X_test, y_train, y_test, results2, drugi)
svc_polymnial(X_train, X_test, y_train, y_test, results2, drugi)
svc_rbf(X_train, X_test, y_train, y_test, results2, drugi)
svc_sigmoid(X_train, X_test, y_train, y_test, results2, drugi)
plots(drugi,X_train,y_train)
plt.clf()
pca = PCA(n_components=2).fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig("Wykres-2.png")
print("Wynik analizy głównych składowych PCA dla dwóch komponentów:")
print(pca.explained_variance_ratio_)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

svc_linear(X_train, X_test, y_train, y_test, results2, drugi)
svc_polymnial(X_train, X_test, y_train, y_test, results2, drugi)
svc_rbf(X_train, X_test, y_train, y_test, results2, drugi)
svc_sigmoid(X_train, X_test, y_train, y_test, results2, drugi)
plots(drugi,X_train,y_train)
