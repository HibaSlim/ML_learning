import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv')

# Display general information about the dataset
print(data.info())
print(data.describe())
print(data.head())

# Generate a profiling report
profile = ProfileReport(data, title="Pandas Profiling Report", explorative=True)
profile.to_file("report.html")

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Remove duplicates if they exist
data.drop_duplicates(inplace=True)

# Handle outliers using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Encode categorical features
data = pd.get_dummies(data, drop_first=True)

# Prepare dataset for modelling
X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with the actual target column name
y = data['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Hyperparameter tuning
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)

# Unsupervised Learning
X_unsupervised = data.drop('target_variable', axis=1)

# Fit KMeans
kmeans = KMeans(n_clusters=3)  # Start with 3 clusters
kmeans.fit(X_unsupervised)
data['cluster'] = kmeans.labels_

# Plot the clusters
sns.scatterplot(x=data['feature1'], y=data['feature2'], hue=data['cluster'])  # Replace 'feature1' and 'feature2' with actual feature names
plt.title('K-Means Clustering')
plt.show()

# Elbow method to find the optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_unsupervised)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.show()
