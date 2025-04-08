import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


#1 -Import you data and perform basic data exploration phase
#     Display general information about the dataset
df = pd.read_csv('African_crises_dataset.csv')

print(df.describe())
print(df.head())
print(df.info())

#    Create a pandas profiling reports to gain insights into the dataset
profile= ProfileReport(df,title= 'African crisis (1860-2014)', explorative= True)
profile.to_file('African_crises.html')

#    Handle Missing and corrupted values
missing_values = df.isnull().sum()
print(f'missing values are: {missing_values[missing_values > 0]}')
# there is no missing values if there was we can drop them or replace them

#    Remove duplicates, if they exist
df.drop_duplicates(inplace=True)

#    Handle outliers, if they exist
#Handle outliers, if they exist
# Visualize outliers using boxplots
plt.figure(figsize=(9, 5))
sns.boxplot(data=df)
plt.title("Africa crisis")
plt.ylabel("Values")
plt.savefig('Africa crisis.png')
#No outliers

#    Encode categorical features
# Identify categorical features
categorical_features = df.select_dtypes(include=['object']).columns
print(categorical_features)

df["banking_crisis"]=df["banking_crisis"].map({"no_crisis": 1, "crisis":2})
df=df.drop(['country_number','country_code'],axis=1)
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])

print(df.info())
print(df.head())

# 2 - Select your target variable and the features
X = df.drop(['systemic_crisis'], axis=1)
y = df['systemic_crisis']
# 3 - Split your dataset to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4 - Based on your data exploration phase select a ML classification algorithm and train it on the training set
# we will use the random forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# 5 - Assess your model performance on the test set using relevant evaluation metrics
y_pred = model.predict(X_test)
print('the prediction are:',y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# 6 - Discuss with your cohort alternative ways to improve your model performance