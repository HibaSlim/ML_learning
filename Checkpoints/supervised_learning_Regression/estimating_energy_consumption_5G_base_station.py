import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats

#load data
df= pd.read_csv('5G_energy_consumption_dataset.csv')
#basic data exploration
print(df.head())
print(df.describe())
print(df.info())

# Create a profile report using pandas profiling
profile = ProfileReport(df, title="5G energy consumption", explorative=True)
profile.to_file("5G_energy_consumption.html")

#to check missing values we use
missing_values = df.isnull().sum()
print(f'missing values ara: {missing_values[missing_values > 0]}')
# there is no missing values if there was we can drop them or replace them

#to remove duplicate we use
df.drop_duplicates(inplace=True)

#Handle outliers, if they exist
# Visualize outliers using boxplots
plt.figure(figsize=(9, 5))
sns.boxplot(data=df)
plt.title("Box Plot using Seaborn")
plt.ylabel("Values")
plt.savefig('boxplot.png')
Q1 = df['Energy'].quantile(0.25)
Q3 = df['Energy'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
filtered_df = df[(df['Energy'] >= lower_bound) & (df['Energy'] <= upper_bound)]
# Visualize filtered data
plt.boxplot(filtered_df['Energy'])
plt.savefig('boxplot_filtered.png')
#trying the z_scores
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df = df[(z_scores < 3).all(axis=1)]
plt.figure(figsize=(9, 5))
sns.boxplot(data=df)
plt.title("Box Plot filtered_with_zscores")
plt.ylabel("Values")
plt.savefig('boxplot_filtered_with_zscores.png')

print(df.info())

# Encode categorical features
# Identify categorical features
categorical_features = df.select_dtypes(include=['object']).columns
print(categorical_features)

# data engineering
#we convert time column to datetime format
df['Time'] = pd.to_datetime(df['Time'])
# Extract features from the 'time' column
df['year'] = df['Time'].dt.year
df['month'] = df['Time'].dt.month
df['day'] = df['Time'].dt.day


# now we drop the original column
df.drop(columns=['Time'], inplace=True)

# we can't use One-hot encoding in the BS column because it has lots of variable
#we will use the labelEncoder()


label_encoder = LabelEncoder()
df['BS'] = label_encoder.fit_transform(df['BS'])

print(df.info())
print(df.head())

#2-Select your target variable and the features
target = 'TXpower'
features = df.drop(columns=[target])
x = features
y = df[target]

#3-Split your dataset to training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#4-Based on your data exploration phase select a ML regression algorithm and train it on the training set
#we will try all the options

multiple_model = LinearRegression()
multiple_model.fit(x_train,y_train)
#5-assess the performance of our model
y_pred = multiple_model.predict(x_test)

mae_m = mean_absolute_error(y_test,y_pred)
mse_m = mean_squared_error(y_pred, y_test)
rmse_m = root_mean_squared_error(y_pred, y_test)
r2_m = r2_score(y_pred=y_pred,y_true=y_test )

print(f'Mae multiple : {mae_m:.2f}')
print(f'Mse multiple: {mse_m:.2f}')
print(f'Rmse multiple: {rmse_m:.2f}')
print(f'r2 multiple: {r2_m:.2f}')

ridge_model = Ridge(alpha=90000)
ridge_model.fit(x_train,y_train)

y_ridge_pred = ridge_model.predict(x_test)
mae_r = mean_absolute_error(y_test,y_ridge_pred)
mse_r = mean_squared_error(y_ridge_pred, y_test)
rmse_r = root_mean_squared_error(y_ridge_pred, y_test)
r2_r = r2_score(y_pred=y_ridge_pred,y_true=y_test )

print(f'Mae ridge: {mae_r:.2f}')
print(f'Mse ridge: {mse_r:.2f}')
print(f'Rmse ridge: {rmse_r:.2f}')
print(f'r2 ridge: {r2_r:.2f}')

Lasso_model = Lasso(alpha=0.50)
Lasso_model.fit(x_train,y_train) # lambda * b1**2

y_lasso_pred = Lasso_model.predict(x_test)
mae_l = mean_absolute_error(y_test,y_lasso_pred)
mse_l = mean_squared_error(y_lasso_pred, y_test)
rmse_l = root_mean_squared_error(y_lasso_pred, y_test)
r2_l = r2_score(y_pred=y_lasso_pred,y_true=y_test )

print(f'Mae Lasso : {mae_l:.2f}')
print(f'Mse Lasso : {mse_l:.2f}')
print(f'Rmse Lasso: {rmse_l:.2f}')
print(f'r2 Lasso: {r2_l:.2f}')

elastic_model = ElasticNet(alpha=1, l1_ratio=0.5,random_state=42)
elastic_model.fit(x_train,y_train)

y_elastic_pred = elastic_model.predict(x_test)
mae_E = mean_absolute_error(y_test,y_elastic_pred)
mse_E = mean_squared_error(y_elastic_pred, y_test)
rmse_E = root_mean_squared_error(y_elastic_pred, y_test)
r2_E = r2_score(y_pred=y_elastic_pred,y_true=y_test )

print(f'Mae Elastic : {mae_E:.2f}')
print(f'Mse Elastic : {mse_E:.2f}')
print(f'Rmse Elastic : {rmse_E:.2f}')
print(f'r2  Elastic: {r2_E:.2f}')

#for better visualisation and analysis i will show the above results in a dataframe
results = {
    'Regression_Model': ['Multiple', 'Ridge', 'Lasso', 'Elastic'],
    'MAE': [mae_m,mae_r, mae_l, mae_E],
    'MSE': [mse_m,mse_r,mse_l, mse_E],
    'RMSE': [rmse_m,rmse_r,rmse_l, rmse_E],
    'R²': [r2_m,r2_r,r2_l, r2_E]
}
results_df = pd.DataFrame(results)
print('\nthe results for each method are:\n',results_df)
print('\nBased on the evaluation metrics the multiple model is the best performing model as it has the lowest MSE,MAE, RMSE,the highest r2')