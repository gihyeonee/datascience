import pandas as pd
import numpy as np
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

df


# Visualization setup
%matplotlib
%config InlineBackend.figure_format = 'svg'

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
plt.ion() # enable the interactive mode

import seaborn as sns
sns.set()  # set plot styles


# Interpolate for entire data
df.interpolate(inplace=True)
df.info()


"""
Correlation Matrix
"""
sns.pairplot(df, kind='reg', diag_kind='kde',
             plot_kws={'scatter_kws': {'alpha': 0.1}})


# Prepare a feature set
X = df.iloc[:, 1:]    # input features
X

y = df.iloc[:, 0]    # target variable
y


"""
Feature Importance
"""

# Create and train model for regression

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)


# Print feature importances
print(model.feature_importances_)


# Plot the feature importances
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
