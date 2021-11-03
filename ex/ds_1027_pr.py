import pandas as pd
import numpy as np

"""
    numpy.concatenate()
"""
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]

print(np.concatenate([x, y, z]), '\n')
print(np.vstack([x, y, z]), '\n')

x = [[1, 2], # 2D array
     [3, 4]]

# axis=0(row-based), 1(column-based)
np.concatenate([x, x], axis=1)


"""
    pandas.concat()
"""

# Simply concatenate the two Series objects
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])

pd.concat([ser1, ser2])


# make_df(): create a DataFrame object

def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind]
           for c in cols}

    return pd.DataFrame(data, ind)

# DataFrame example
print(make_df('ABC', [1, 2]))


# Concatenate the two DataFrame objects

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])

print(df1, '\n')
print(df2, '\n')

print(pd.concat([df1, df2], axis=1)) # default axis is 0.


# Change the axis to 0.

df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])

print(df3, '\n')
print(df4, '\n')

print(pd.concat([df3, df4], axis=1))


"""
    concate() Features
"""

# Case 1: Duplicate indices

x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])

y.index = x.index # make duplicate indices!

print(x, '\n')
print(y, '\n')

print(pd.concat([x, y]))


# Case 2: Catching duplications as errors

try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError: ", e)



# Case 3: Ignore the index:
# assign integer-based indices automatically

pd.concat([x, y], ignore_index=True)



# Case 4: Adding multiindex keys (hierarchical indexing)

pd.concat([x, y], keys=['x', 'y'])



"""
    concat(): join options
"""

x = make_df('ABC', [1, 2])
y = make_df('BCD', [3, 4])

# default: outer join -> union

print(x, '\n')
print(y, '\n')
print(pd.concat([x, y], join='outer'))


# inner join -> intersection

print(df5, '\n')
print(df6, '\n')
print(pd.concat([df5, df6], join='inner'))


# append()

print(df1); print()
print(df2); print()
print(df1.append(df2))


"""
    Join operations
"""

# Prepare the two DataFrame objects
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                   'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})

df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                   'hire_date': [2004, 2008, 2012, 2014]})

print(df1, '\n')
print(df2)


# One-to-one joins

df3 = pd.merge(df1, df2)

print(df3)


# Many-to-one joins

df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                   'supervisor': ['Carly', 'Guido', 'Steve']})

print(df3, '\n')
print(df4, '\n')

print(pd.merge(df3, df4))


# Many-to-many joins

df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                             'Engineering', 'Engineering',
                             'HR', 'HR'],
                   'skills': ['math', 'spreadsheets', 'coding',
                             'linux', 'spreadsheets', 'organization']})

print(df1, '\n')
print(df5, '\n')

print(pd.merge(df1, df5))


"""
    Specifying the merge Key
"""

# 'on' argument: use a common column as merge key

print(df1, '\n')
print(df2, '\n')

df1['age'] = pd.Series([20, 42, 37, 25])
df2['age'] = pd.Series([37, 20, 42, 25])
print(df1, '\n')
print(df2, '\n')

print(pd.merge(df1, df2, on='employee'), '\n')


# 'left_on' and 'right_on' arguments:
# merge two datasets with different columns(e.g., employee and name)

df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                   'salary': [70000, 80000, 120000, 90000]})

print(df1, '\n')
print(df3, '\n')


pd.merge(df1, df3, left_on="employee", right_on="name")


# 'left_index' and 'right_index' arguments
# merge on and index ('employee')

df1a = df1.set_index('employee')
df2a = df2.set_index('employee')

print(df1a, '\n\n', df2a)


pd.merge(df1a, df2a, left_index=True, right_index=True)


"""
    Specifying the Join method
"""

df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                   'food': ['fish', 'beans', 'bread']},
                  columns=['name', 'food'])

df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                   'drink': ['wine', 'beer']},
                  columns=['name', 'drink'])

print(df6, '\n')
print(df7)


# inner join (default)

pd.merge(df6, df7, how='inner')


# outer join

pd.merge(df6, df7, how='outer')


# left join

pd.merge(df6, df7, how='left')


# right join

pd.merge(df6, df7, how='right')
