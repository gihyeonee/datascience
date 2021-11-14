import numpy as np
import pandas as pd

"""
None: Pythonic missing data
"""

# None objects as missing values
arr1 = np.array([1, None, 3, 4])
print(arr1.dtype)

arr2 = np.array([1, 2, 3, 4])
print(arr2.dtype)


# Python objects are incompatible with numpy and pandas operations
arr1.sum()

arr2.sum()


"""
NaN: Missing Numerical Data
"""

arr3 = np.array([1, np.nan, 3, 4])
print(arr3.dtype)


# Arithmetic with NaN will be another NaN

print(1 + np.nan)
print(0 * np.nan)
print(arr3.sum())


# Special NumPy aggregation funcs that ignore these missing values
print(np.nansum(arr3))
print(np.nanmax(arr3))
print(np.nanmin(arr3))


# Pandas automatically converts the None to a NaN value.

pd.Series([1, np.nan, 2, None])


"""
Detecting null values
"""

# isnull()

ser = pd.Series([1, np.nan, 'hello', None])
ser.isnull()


# notnull()

ser.notnull()


"""
Dropping null values
"""

# dropna()
ser.dropna()


# For a DataFrame, there are more options

df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df


# df.dropna(): list-wise deletion
df.dropna()


# df.dropna(axis='columns'): variable deletion
df.dropna(axis='columns')


# how/thresh parameters
df[3] = np.nan
df


# how='any' (default)
# how='all' which will only drop rows/columns that are all null values
df.dropna(axis='columns', how='all')


# thresh = minimum number of non-null values to be kept
df.dropna(axis='rows', thresh=3)


"""
Filling Null Values
"""

ser = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
ser

# Fill null values with a certain value
ser.fillna(0)


# Forward-fill = LOCF
ser.fillna(method='ffill') # equals to "ser.ffill()"


# backward-fill = NOCB
ser.fillna(method='bfill') # equals to "ser.bfill()"


# bfill with rows
df.fillna(method='bfill', axis='rows')


from pandas import datetime
from matplotlib import pyplot as plt

"""
Load AirQualityUCI Data
"""

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

input_file = './data/AirQualityUCI_refined.csv'

df = pd.read_csv(input_file,
                 index_col=[0],
                 parse_dates=[0],
                 date_parser=parser)


# Print the summary of the dataset

df.head()
df.info()


# Visualization setup
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # set plot styles
%config InlineBackend.figure_format = 'svg'


# Visualize the series of CO(GT)

df['CO(GT)'].plot()


# imputation

imp_locf = df['CO(GT)'].copy().ffill()
imp_nocb = df['CO(GT)'].copy().bfill()
imp_linear = df['CO(GT)'].copy().interpolate()
imp_mean = df['CO(GT)'].copy().fillna(df['CO(GT)'].mean())


# k-nn imputation

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)    # default: 2
imp_knn = df.copy().values
imp_knn = imputer.fit_transform(imp_knn)


# add indices to the imputed result of k-nn

imp_df = pd.DataFrame(imp_knn, index=imp_locf.index, columns=df.columns)


# Visualizing the imputed results

plt.plot(df['CO(GT)'], label='actual', zorder=10)
plt.plot(imp_locf, label='locf', zorder=1)
plt.plot(imp_nocb, label='nocb', zorder=2)
plt.plot(imp_linear, label='linear interpolation', zorder=3)
plt.plot(imp_mean, label='mean substitution', zorder=4)
plt.plot(imp_df['CO(GT)'], label='k-nearest neighbor', zorder=5)
plt.legend(loc='best')
plt.show()


# Select the certain period to visualize

start = '2004-07-18'
end = '2004-10-20'

# Visualize 2004-07 ~ 2004-10

plt.plot(df['CO(GT)'].loc[start:end], label='actual', zorder=10)
plt.plot(imp_locf.loc[start:end], label='locf', zorder=1)
plt.plot(imp_nocb.loc[start:end], label='nocb', zorder=2)
plt.plot(imp_linear.loc[start:end], label='linear interpolation', zorder=3)
plt.plot(imp_mean.loc[start:end], label='mean substitution', zorder=4)
plt.plot(imp_df['CO(GT)'].loc[start:end], label='k-nearest neighbor', zorder=5)
plt.legend(loc='best')
plt.show()
