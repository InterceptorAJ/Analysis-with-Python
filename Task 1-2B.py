import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# dataset import
dataset = pd.read_csv (r'manaus.csv')

#check for null records in dataset
checknulls = dataset['manaus'].isnull().any()
manaus = dataset['manaus']
group = manaus.groupby(dataset['time'])


#check for minimal value - if there is 0
count = dataset['manaus'].min()

#generate sample data no1
num_samples1 = 20
desired_mean = 0.0
desired_std_dev = 2.0

samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=num_samples1)

actual_mean = np.mean(samples)
actual_std = np.std(samples)
zero_mean_samples = samples - (actual_mean)
zero_mean_mean = np.mean(zero_mean_samples)
zero_mean_std = np.std(zero_mean_samples)
scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std)
scaled_mean = np.mean(scaled_samples)
scaled_std = np.std(scaled_samples)
final_samples = scaled_samples + desired_mean
final_mean = np.mean(final_samples)
final_std = np.std(final_samples)

#generate sample data no2
num_samples2 = 15
desired_mean2 = 0.0
desired_std_dev2 = 3.0

samples2 = np.random.normal(loc=0.0, scale=desired_std_dev2, size=num_samples2)
actual_mean2 = np.mean(samples2)
actual_std2 = np.std(samples2)
zero_mean_samples2 = samples2 - (actual_mean2)
zero_mean_mean2 = np.mean(zero_mean_samples2)
zero_mean_std2 = np.std(zero_mean_samples2)
scaled_samples2 = zero_mean_samples2 * (desired_std_dev2/zero_mean_std2)
scaled_mean2 = np.mean(scaled_samples2)
scaled_std2 = np.std(scaled_samples2)
final_samples2 = scaled_samples2 + desired_mean2
final_mean2 = np.mean(final_samples2)
final_std2 = np.std(final_samples2)
print (final_mean)

#t-student test for the same:
all_manaus = dataset['manaus']
sample_manaus1 = dataset['manaus'].sample(20)
sample_manaus2 = dataset['manaus'].sample(15)
sample_manaus = np.concatenate((sample_manaus1, sample_manaus2))
stats1 = stats.ttest_1samp(a=sample_manaus, popmean=all_manaus.mean())
res = stats1.pvalue

#t-student test for new sample:
all_manaus = dataset['manaus']
sample_manaus1_new = final_samples
sample_manaus2_new = final_samples2
sample_manaus_new = np.concatenate((final_samples, final_samples2))
stats2 = stats.ttest_1samp(a=sample_manaus_new, popmean=all_manaus.mean())
res1 = stats2.pvalue

#plot hiatogram of all population
manaus_all_hist = dataset.hist(column='manaus', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
manaus_all_hist = manaus_all_hist[0]
for x in manaus_all_hist:

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
    x.set_xlabel("Wysokość rzeki", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([0, 0], [0, 150], 'k-', lw=2)

# convert array to dataframe first sample
sample_manaus_df = pd.DataFrame(np.array(sample_manaus).reshape(-1,1), columns = list("m"))

#plot hiatogram of first sample
sample_manaus_df_hist = sample_manaus_df.hist(column='m', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
sample_manaus_df_hist = sample_manaus_df_hist[0]
for y in sample_manaus_df_hist:

    # Despine
    y.spines['right'].set_visible(False)
    y.spines['top'].set_visible(False)
    y.spines['left'].set_visible(False)

    # Switch off ticks
    y.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = y.get_yticks()
    for tick in vals:
        y.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    y.set_title("")

    # Set x-axis label
    y.set_xlabel("Wysokość rzeki dla próbki nr1", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    y.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    y.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([0, 0], [0, 6], 'k-', lw=2)

# convert array to dataframe second sample
sample_manaus_new_df = pd.DataFrame(np.array(sample_manaus_new).reshape(-1,1), columns = list("b"))

#plot hiatogram of second sample
sample_manaus_new_df_hist = sample_manaus_new_df.hist(column='b', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
sample_manaus_new_df_hist = sample_manaus_new_df_hist[0]
for z in sample_manaus_new_df_hist:

    # Despine
    z.spines['right'].set_visible(False)
    z.spines['top'].set_visible(True)
    z.spines['left'].set_visible(False)

    # Switch off ticks
    z.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = z.get_yticks()
    for tick in vals:
        z.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    z.set_title("")

    # Set x-axis label
    z.set_xlabel("Wysokość rzeki dla próbki nr2", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    z.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    z.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([0, 0], [0, 6], 'k-', lw=2)

print ('---------------------------------------------------------------------------------------------------------------------------------')
print ('Ilość wierszy i kolumn: ')
print (dataset.shape)
print ('---------------------------------------------------------------------------------------------------------------------------------')
print ("Czy występują puste wartości w polu Manaus? ")
if checknulls == False:
    print ("Brak wartości pustych!")
else:
    print ('Występują wartości puste!')
    exit()
print ()
print('---------------------------------------------------------------------------------------------------------------------------------')
print ('Czy występują zerowe lub ujemne wartości w polu Manaus?')
if count > 0:
    print ("Najniższa wartość jest dodatnia i wynosi: " + str(count))
elif count == 0:
    print ("Występuje wartość zerowa!")
    exit()
else:
    print ("Występuje wartość ujemna!")
print ()
print('---------------------------------------------------------------------------------------------------------------------------------')
print ('Sprawdzamy średnią próby niezależnej i porównujemy ze średnią w populacji: ')
print ('Średnia wartość dla populacji: ' + '%.8f' % float(all_manaus.mean()))
print ('Srednia wartość dla próby: ' + '%.8f' % float(sample_manaus_new.mean()))
# print (stats2)
print ('pvalue :' '%.8f' % res1)
if res1 > 0.05:
    print ('Rozkłady są zgodne! - nie ma podstaw do odrzucenia Hipotezy')
else:
    print ('Istotne różnice statystyczne! - Hipoteza odrzucona')
print('---------------------------------------------------------------------------------------------------------------------------------')
print ('Sprawdzamy średnią próby zależnej i porównujemy ze średnią w populacji: ')
print ('Średnia wartość dla populacji: ' + '%.8f' % float(all_manaus.mean()))
print ('Średnia wartość dla próby: ' + '%.8f' % float(sample_manaus.mean()))
# print (stats1)
print ('pvalue :' '%.8f' % res)
if res > 0.05:
    print ('Rozkłady są zgodne! - nie ma podstaw do odrzucenia Hipotezy')
else:
    print ('Istotne różnice statystyczne! - Hipoteza odrzucona')
print('---------------------------------------------------------------------------------------------------------------------------------')

plt.show()
