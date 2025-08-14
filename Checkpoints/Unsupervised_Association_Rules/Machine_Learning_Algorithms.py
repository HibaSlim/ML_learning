import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Microsoft_malware_dataset_min.csv')

# Basic Data Exploration
print(data.info())
print(data.describe())

# Create a Pandas Profiling Report
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
profile.to_file("data_profile_report.html")

# Handle Missing Values
data.fillna(data.mean(), inplace=True)

# Remove Duplicates
data.drop_duplicates(inplace=True)

# Handle Outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Encode Categorical Features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Prepare Dataset for Modelling
X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with your actual target column
y = data['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Decision Tree and Plot ROC Curve
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Improve Model Performance by Changing Hyperparameters
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)

# Unsupervised Learning
X_unsupervised = data.drop('target_variable', axis=1)  # Replace 'target_variable' with your actual target column

# Apply K-Means Clustering and Plot the Clusters
kmeans = KMeans(n_clusters=3)  # Start with 3 clusters
kmeans.fit(X_unsupervised)
data['cluster'] = kmeans.labels_

plt.scatter(data['feature1'], data['feature2'], c=data['cluster'], cmap='viridis')  # Replace 'feature1' and 'feature2'
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Find the Optimal K Parameter
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_unsupervised)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
