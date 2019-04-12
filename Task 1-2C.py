import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# dataset import
dataset = pd.read_csv (r'quakes.csv')

#check for null records in dataset
checknulls = dataset['depth'].isnull().any()
quakes = dataset['depth']
group = quakes.groupby(dataset['depth'])


#check for minimal value - if there is 0
count = dataset['depth'].min()

#generate sample data no1
num_samples1 = 25
desired_mean = 300
desired_std_dev = 15

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
num_samples2 = 10
desired_mean2 = 300
desired_std_dev2 = 15

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
all_quakes = dataset['depth']
sample_quakes1 = dataset['depth'].sample(20)
sample_quakes2 = dataset['depth'].sample(15)
sample_quakes = np.concatenate((sample_quakes1, sample_quakes2))
stats1 = stats.ttest_1samp(a=sample_quakes, popmean=all_quakes.mean())
res = stats1.pvalue

#t-student test for new sample:
all_quakes = dataset['depth']
sample_quakes1_new = final_samples
sample_quakes2_new = final_samples2
sample_quakes_new = np.concatenate((final_samples, final_samples2))
stats2 = stats.ttest_1samp(a=sample_quakes_new, popmean=all_quakes.mean())
res1 = stats2.pvalue

#plot hiatogram of all population
quakes_all_hist = dataset.hist(column='depth', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
quakes_all_hist = quakes_all_hist[0]
for x in quakes_all_hist:

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
    x.set_xlabel("Głębokość Trzęsienia Ziemi", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([300, 300], [0, 150], 'k-', lw=2)

# convert array to dataframe - first sample
sample_quakes_df = pd.DataFrame(np.array(sample_quakes).reshape(-1,1), columns = list("q"))

#plot hiatogram of first sample
sample_quakes_df_hist = sample_quakes_df.hist(column='q', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
sample_quakes_df_hist = sample_quakes_df_hist[0]
for y in sample_quakes_df_hist:

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
    y.set_xlabel("Głębokość Trzęsienia Ziemi dla próbki nr1", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    y.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    y.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([300, 300], [0, 6], 'k-', lw=2)

# convert array to dataframe - second sample
sample_quakes_new_df = pd.DataFrame(np.array(sample_quakes_new).reshape(-1,1), columns = list("t"))

#plot hiatogram of second sample
sample_quakes_new_df_hist = sample_quakes_new_df.hist(column='t', bins=25, grid="false", figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
sample_quakes_new_df_hist = sample_quakes_new_df_hist[0]
for z in sample_quakes_new_df_hist:

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
    z.set_xlabel("Głębokość Trzęsienia Ziemi dla próbki nr2", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    z.set_ylabel("Liczba wystąpień", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    z.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    plt.plot([300, 300], [0, 6], 'k-', lw=2)

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
print ('Średnia wartość dla populacji: ' + '%.8f' % float(all_quakes.mean()))
print ('Średnia wartość dla próby: ' + '%.8f' % float(sample_quakes_new.mean()))
# print (stats2)
print ('pvalue :' '%.8f' % res1)
if res1 > 0.05:
    print ('Rozkłady są zgodne! - nie ma podstaw do odrzucenia Hipotezy')
else:
    print ('Istotne różnice statystyczne! - Hipoteza odrzucona')
print('---------------------------------------------------------------------------------------------------------------------------------')
print ('Sprawdzamy średnią próby zależnej i porównujemy ze średnią w populacji: ')
print ('Średnia wartość dla populacji: ' + '%.8f' % float(all_quakes.mean()))
print ('Średnia wartość dla próby: ' + '%.8f' % float(sample_quakes.mean()))
# print (stats1)
print ('pvalue :' '%.8f' % res)
if res > 0.05:
    print ('Rozkłady są zgodne! - nie ma podstaw do odrzucenia Hipotezy')
else:
    print ('Istotne różnice statystyczne! - Hipoteza odrzucona')
print('---------------------------------------------------------------------------------------------------------------------------------')
plt.show()
