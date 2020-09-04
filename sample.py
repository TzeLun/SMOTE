import pandas as pd
import SMOTE as sm

df = pd.read_csv('sample.csv', header=None)


# Simple pre-processing function to get the desired format for the dataset
def pre_processing(dataset):
    d = dataset.T
    return [list(d[i]) for i in d]


df = pre_processing(df)
minority = df[50:75]  # Use all 25 class 'B' data as the input dataset

# The SMOTE function is labelled as augment()
syn = sm.augment(minority, 50, 5)
print(syn)
print(len(syn))

# syn = sm.augment(minority, 100, 7)
# print(syn[0:25])
# print(syn[25:50])
