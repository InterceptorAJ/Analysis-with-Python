import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.pyplot import hist

# dataset import
dataset = pd.read_csv (r'iris.csv')

# setting up columns as variables
A = dataset['sepal length']
B = dataset['sepal width']
C = dataset['petal length']
D = dataset['petal width']
E = dataset['class']

# setting up variables for maximum values
max1 = A.max()
max2 = B.max()
max3 = C.max()
max4 = D.max()

# setting up variables for minimum values
min1 = A.min()
min2 = B.min()
min3 = C.min()
min4 = D.min()

# setting up variables for median values
median1 = A.median()
median2 = B.median()
median3 = C.median()
median4 = D.median()

# setting up variable for no of counted values
dom = E.value_counts()

# correlation
df = pd.DataFrame(data = dataset)

def get_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, 3):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[2:n]

#correlation plots
petallhist = dataset.hist(column='petal length', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
petallhist = petallhist[0]
for x in petallhist:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    x.set_title("")

    # Set x-axis label
    x.set_xlabel("Długość płatka", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

petalwhist = dataset.hist(column='petal width', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
petalwhist = petalwhist[0]
for x in petalwhist:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    x.set_title("")

    # Set x-axis label
    x.set_xlabel("Szerokość płatka", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

#petalwhist = dataset.hist(column='petal width', bins=30)
#correlation no2
#setallhist = dataset.hist(column='sepal length', bins=30)
#setalplhist = dataset.hist(column='petal length', bins=30)

print ('---------------------------------------------------------------------------------------------------------------------------------')
print ('---------- Wartości maksymalne:')
print ('---------- Max sepal length: ' + str(max1),'Max sepal width: ' + str(max2),'Max petal length: ' + str(max3),'Max petal width: ' + str(max4))
print()
print ('---------------------------------------------------------------------------------------------------------------------------------')
print ('---------- Wartości minimalne:')
print ('---------- Min sepal length: ' + str(min1),'Min sepal width: ' + str(min2),'Min petal length: ' + str(min3),'Min petal width: ' + str(min4))
print()
print ('---------------------------------------------------------------------------------------------------------------------------------')
print ('---------- Wartości środkowe:')
print ('---------- Mediana sepal length: ' + str(median1),'Mediana sepal width: ' + str(median2),'Mediana petal length: ' + str(median3),'Mediana petal width: ' + str(median4))
print()
print ('---------------------------------------------------------------------------------------------------------------------------------')
print ('---------- Częstotliwość występowania (malejąco):')
print (dom)
print()
print ('---------------------------------------------------------------------------------------------------------------------------------')
print("---------- Macierz korelacji")
print(df.corr())
print()
print ('---------------------------------------------------------------------------------------------------------------------------------')
print("---------- Lista korelacji (malejąco):")
print(get_top_correlations(df, 4))
print()
plt.show()
