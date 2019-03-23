import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# dataset import
dataset = pd.read_csv (r'Births.csv')

#check for null records in dataset
checknulls = dataset['births'].isnull().any()
births = dataset['births']
group = births.groupby(dataset['date'])


#check for minimal value - if there is 0
count = dataset['births'].min()

#generate sample data no1
num_samples1 = 30
desired_mean = 10000.0
desired_std_dev = 500.0

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
num_samples2 = 20
desired_mean2 = 10000.0
desired_std_dev2 = 700.0

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

#t-student test for dependent values:
population_births = dataset['births']
sample_births1 = dataset['births'].sample(30)
sample_births2 = dataset['births'].sample(20)
sample_births = np.concatenate((sample_births1, sample_births2))
stats1 = stats.ttest_1samp(a=sample_births, popmean=population_births.mean())
res = stats1.pvalue

#t-student test for independent values (new sample):
population_births = dataset['births']
sample_births1_new = final_samples
sample_births2_new = final_samples2
sample_births_new = np.concatenate((final_samples, final_samples2))
stats2 = stats.ttest_1samp(a=sample_births_new, popmean=population_births.mean())
res1 = stats2.pvalue

#plot hiatogram of all population
births_pop_hist = dataset.hist(column='births', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
births_pop_hist = births_pop_hist[0]
for x in births_pop_hist:

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
    x.set_xlabel("Liczba urodzin dla całej populacj", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([10000, 10000], [0, 700], 'k-', lw=2)

# convert array to dataframe first sample
sample_births_df = pd.DataFrame(np.array(sample_births).reshape(-1,1), columns = list("b"))

#plot hiatogram of first sample
sample_births_df_hist = sample_births_df.hist(column='b', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
sample_births_df_hist = sample_births_df_hist[0]
for y in sample_births_df_hist:

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
    y.set_xlabel("Liczba urodzin dla próbki nr1 (próbka zależna)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    y.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    y.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([10000, 10000], [0, 8], 'k-', lw=2)

# convert array to dataframe second sample
sample_births_new_df = pd.DataFrame(np.array(sample_births_new).reshape(-1,1), columns = list("b"))

#plot hiatogram of second sample
sample_births_new_df_hist = sample_births_new_df.hist(column='b', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
sample_births_new_df_hist = sample_births_new_df_hist[0]
for z in sample_births_new_df_hist:

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
    z.set_xlabel("Liczba urodzin dla próbki nr2 (próbka niezależna)", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    z.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    z.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([10000, 10000], [0, 8], 'k-', lw=2)

print ('---------------------------------------------------------------------------------------------------------------------------------')
print ('Ilość wierszy i kolumn: ')
print (dataset.shape)
print ('---------------------------------------------------------------------------------------------------------------------------------')
print ("Czy występują puste wartości w polu Births? ")
if checknulls == False:
    print ("Brak wartości pustych!")
else:
    print ('Występują wartości puste!')
    exit()
print ()
print('---------------------------------------------------------------------------------------------------------------------------------')
print ('Czy występują zerowe wartości w polu Births?')
if count > 0:
    print ("Najniższa wartość jest dodatnia i wynosi: " + str(count))
elif count == 0:
    print ("Występuje wartość zerowa!")
    exit()
else:
    print ("Występuje wartość ujemna!")
    exit()
print ()
print('---------------------------------------------------------------------------------------------------------------------------------')
print ('Sprawdzamy średnią próby niezależnej i porównujemy ze średnią w populacji: ')
print ('Średnia wartość dla populacji: ' + '%.8f' % float(population_births.mean()))
print ('Średnia wartość dla próby: ' + '%.8f' % float(sample_births_new.mean()))
# print (stats2)
print ('pvalue :' '%.8f' % res1)
if res1 > 0.05:
    print ('Rozkłady są zgodne! - nie ma podstaw do odrzucenia Hipotezy')
else:
    print ('Istotne różnice statystyczne! - Hipoteza odrzucona')
print('---------------------------------------------------------------------------------------------------------------------------------')
print ('Sprawdzamy średnią próby zależnej i porównujemy ze średnią w populacji: ')
print ('Średnia wartość dla populacji: ' + '%.8f' % float(population_births.mean()))
print ('Średnia wartość dla próby: ' + '%.8f' % float(sample_births.mean()))
# print (stats1)
print ('pvalue :' '%.8f' % res)
if res > 0.05:
    print ('Rozkłady są zgodne! - nie ma podstaw do odrzucenia Hipotezy')
else:
    print ('Istotne różnice statystyczne! - Hipoteza odrzucona')
print('---------------------------------------------------------------------------------------------------------------------------------')
plt.show()
