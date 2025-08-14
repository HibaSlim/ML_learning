import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from pandas_profiling import ProfileReport

# Load the dataset
url = 'path_to_your_dataset.csv'  # Replace with the actual path to your dataset
data = pd.read_csv(url)

# Display general information
print(data.info())
print(data.describe())

# Create a profiling report
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
profile.to_file("profiling_report.html")

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])
data.fillna(data.mean(), inplace=True)

# Remove duplicates
data.drop_duplicates(inplace=True)

# Visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data['price'])  # Replace 'price' with the actual target variable name
plt.show()

# Remove outliers based on IQR
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['price'] >= (Q1 - 1.5 * IQR)) & (data['price'] <= (Q3 + 1.5 * IQR))]

# Encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Define target variable and features
X = data.drop('price', axis=1)  # Replace 'price' with the actual target variable name
y = data['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the SVM model
model = SVR(kernel='linear')  # You can experiment with different kernels
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
