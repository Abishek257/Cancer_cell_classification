# Importing neccessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

# Import the dataset
data = pd.read_csv("cancer_classification.csv")
data.head()

# Renaming the Target column
data.rename(columns={'benign_0__mal_1': 'status'}, inplace=True)

# Info of dataset
data.info()

# Statiscal measures of data
data.describe()

# Data Cleaning
# Checking for missing values
data.isnull().sum()

# Checking for duplicates
data.duplicated().sum()

# Check the data is balanced or imbalanced
# Count of the status
Target_count = data['status'].value_counts()
plt.pie(Target_count, labels=Target_count.index, autopct='%1.2f%%') # Here, the minority class is more than 30%. So, it is balanced.

# Split the features and target
X = data.drop(columns='status')     # features
y = data['status']                  # target

# Split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialization of different classification models
model_nb = GaussianNB()
model_lr = LogisticRegression()
model_rf = RandomForestClassifier()
model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_svc = SVC(kernel='rbf')
model_lgbm = LGBMClassifier()

# Training the models
model_lr.fit(X_train, y_train)
model_nb.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_svc.fit(X_train, y_train)
model_lgbm.fit(X_train, y_train)

# Predicting for the test set with all the models
y_pred_nb = model_nb.predict(X_test)
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_xgb = model_xgb.predict(X_test)
y_pred_svc = model_svc.predict(X_test)
y_pred_lgbm = model_lgbm.predict(X_test)

# Performance of all the classification model
accuracy = accuracy_score(y_test, y_pred_nb)
print(f"Model Accuracy of Naive Bayes: {accuracy * 100:.2f}%")
accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Model Accuracy of Logistic Regression: {accuracy * 100:.2f}%")
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Model Accuracy of Random Forest: {accuracy * 100:.2f}%")
accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"Model Accuracy of XGBoost: {accuracy * 100:.2f}%")
accuracy = accuracy_score(y_test, y_pred_svc)
print(f"Model Accuracy of Support vector classifier: {accuracy * 100:.2f}%")
accuracy = accuracy_score(y_test, y_pred_lgbm)
print(f"Model Accuracy of light GBM: {accuracy * 100:.2f}%")