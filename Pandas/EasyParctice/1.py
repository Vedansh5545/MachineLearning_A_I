import pandas as pd
import numpy as np

df = pd.read_csv('large_data.csv')
df2 = pd.read_csv('Book1.csv')
d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)


print(df.describe())
print(df2.describe())
