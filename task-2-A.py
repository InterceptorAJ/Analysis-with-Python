import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import linregress
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# dataset import
from sklearn.preprocessing import Imputer

df = pd.read_csv("data_banknote_authentication.txt", names= ["variance", "skewnes", "curtosis", "entropy", "class"])
print(df.describe())

df[['variance','skewnes','curtosis','entropy']] = df[['variance','skewnes','curtosis','entropy']].replace(0, np.NaN)
print(df.describe())
# dataset null values check
print('-------------------------------------------------------------------------------------------------------------')
print('Ilość pustych danych: ')
a = (df.isnull().sum())
print(a)
print('-------------------------------------------------------------------------------------------------------------')
b = df.shape
b = (b[0]-1)
a = sum(a)
c = a / b * 100
print('Wszystkich danych: ',b)
print('Pustych danych: ', a)
print('Procentowy udział pustych danych: ', round(c, 2), '%')

df = pd.read_csv("data_banknote_authentication.txt", names= ["variance", "skewnes", "curtosis", "entropy", "class"])
df[['variance','skewnes','curtosis','entropy']] = df[['variance','skewnes','curtosis','entropy']].replace(0, np.NaN)
df_desc = df.describe()
# remove null values for df1
df1 = df
df1.dropna(inplace=True)
df1_desc = df1.describe()

values = df1.values
X = values[:,0:4]
y = values[:,4]
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print('Współczynnik LDA dla danych bez pustych danych: ', result.mean())

def plot_regression(argument1, argument2):
    x = dframe[argument1]
    y = dframe[argument2]

    stats = linregress(x, y)

    m = stats.slope
    b = stats.intercept

    # Change the default figure size
    plt.figure(figsize=(10,10))

    # Change the default marker for the scatter from circles to x's
    plt.scatter(x, y, marker='x')

    # Set the linewidth on the regression line to 3px
    plt.plot(x, m * x + b, color="red", linewidth=3)

    # Add x and y lables, and set their font size
    plt.xlabel(argument1, fontsize=20)
    plt.ylabel(argument2, fontsize=20)

    # Set the font size of the number lables on the axes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig(argument1+argument2+"added.png")

#draws plots with deleted values - run once only!
# plot_regression('variance', 'skewnes', dframe)
# plot_regression('variance', 'curtosis', dframe)
# plot_regression('variance', 'entropy', dframe)
# plot_regression('skewnes', 'curtosis', dframe)
# plot_regression('skewnes', 'entropy', dframe)
# plot_regression('curtosis', 'entropy', dframe)

print('-------------------------------------------------------------------------------------------------------------')
print('Dane pzred usunięciem pustych wartości: ')
print(df_desc)
print('-------------------------------------------------------------------------------------------------------------')
print('Dane po usunięciu pustych wartości: ')
print(df1_desc)

# data clearing and mean imputing
dframe = pd.read_csv("data_banknote_authentication.txt", names= ["variance", "skewnes", "curtosis", "entropy", "class"])
dframe[['variance','skewnes','curtosis','entropy']] = dframe[['variance','skewnes','curtosis','entropy']].replace(0, np.NaN)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(dframe)
dframe = np.array(imp_mean.transform(dframe))
dframe = pd.DataFrame({'variance':dframe[:,0],'skewnes':dframe[:,1],'curtosis':dframe[:,2],'entropy':dframe[:,3],'class':dframe[:,4]})
print('-------------------------------------------------------------------------------------------------------------')
print('Dane po dodaniu wartości metodą imput mean: ')
print(dframe.describe())

#draws plots with added(mean) values - run once only!
# plot_regression('variance', 'skewnes')
# plot_regression('variance', 'curtosis')
# plot_regression('variance', 'entropy')
# plot_regression('skewnes', 'curtosis')
# plot_regression('skewnes', 'entropy')
# plot_regression('curtosis', 'entropy')

# interpolation method
df = pd.read_csv("data_banknote_authentication.txt", names= ["variance", "skewnes", "curtosis", "entropy", "class"])
df[['variance','skewnes','curtosis','entropy']] = df [['variance','skewnes','curtosis','entropy']].replace(0, np.NaN)
print('-------------------------------------------------------------------------------------------------------------')
print('Dane po dodaniu wartości metodą interpolacji: ')
df_interpolate = df.interpolate()
print(df_interpolate.describe())






