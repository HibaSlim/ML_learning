import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

#load data
df = pd.read_csv('Credit_card_dataset.csv')
#basic data exploration
print(df.head())
print(df.describe())
print(df.info())

#Perform the necessary data preparation steps

#missing values
missing_values = df.isnull().sum()
print(f'missing values are: {missing_values[missing_values > 0]}')

#removing duplicate
df.drop_duplicates(inplace=True)

# Encode categorical features
label_encoder = LabelEncoder()
df['CUST_ID'] = label_encoder.fit_transform(df['CUST_ID'])

print(df.info())
print(df.head())

# outliers visualization and handling
plt.figure(figsize=(9, 5))
sns.boxplot(data=df)
plt.title("Before removing outliers using Seaborn")
plt.savefig('Before removing outliers.png')

columns_with_outliers = ['PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT','CASH_ADVANCE']
z_scores = np.abs(stats.zscore(df[columns_with_outliers]))
df_no_outliers = df[(z_scores < 3).all(axis=1)]
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_no_outliers)
plt.title("after removing outliers with z_scores")
plt.savefig('after removing outliers with z_''scores.png')

Q1 = df[columns_with_outliers].quantile(0.25)
Q3 = df[columns_with_outliers].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
filtered_df = df[(df[columns_with_outliers] >= lower_bound) & (df[columns_with_outliers] <= upper_bound)]
# Visualize filtered data
plt.boxplot(filtered_df[columns_with_outliers])
plt.title("no outliers using lower bound and the upper bound methods")
plt.savefig('no outliers using lower bound and the upper bound methods.png')
print('original\n',df.info())
print('zscore\n',df_no_outliers.info())
print(df_no_outliers.head())
print('bound\n',filtered_df.info())