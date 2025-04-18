
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# --- Data Processing ---
# 1. Handling Missing Values
df = df.dropna(subset=['TotalCharges'])

# 2. Encoding Categorical Variables
# a. Binary variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Churn'] = label_encoder.fit_transform(df['Churn'])
df['Partner'] = label_encoder.fit_transform(df['Partner'])
df['Dependents'] = label_encoder.fit_transform(df['Dependents'])
df['PhoneService'] = label_encoder.fit_transform(df['PhoneService'])
df['OnlineSecurity'] = label_encoder.fit_transform(df['OnlineSecurity'])
df['OnlineBackup'] = label_encoder.fit_transform(df['OnlineBackup'])
df['DeviceProtection'] = label_encoder.fit_transform(df['DeviceProtection'])
df['TechSupport'] = label_encoder.fit_transform(df['TechSupport'])
df['StreamingTV'] = label_encoder.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = label_encoder.fit_transform(df['StreamingMovies'])
df['MultipleLines'] = label_encoder.fit_transform(df['MultipleLines'])
df['PaperlessBilling'] = label_encoder.fit_transform(df['PaperlessBilling'])

# b. One-Hot Encoding for multi-class variables
ct = ColumnTransformer(
    [('one_hot', OneHotEncoder(handle_unknown='ignore'), ['Contract', 'PaymentMethod', 'InternetService'])],
    remainder='passthrough'
)
encoded_data = ct.fit_transform(df)
encoded_df = pd.DataFrame(encoded_data)
encoded_df.columns = ct.get_feature_names_out()

# 3. Standardizing Numerical Features
numerical_features = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Gender', 'Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'PaperlessBilling']
scaler = StandardScaler()
encoded_df[numerical_features] = scaler.fit_transform(encoded_df[numerical_features])

# 4. Feature Selection (Dropping CustomerID)
encoded_df = encoded_df.drop('remainder__customerID', axis=1)

# Prepare data for modeling
X = encoded_df.drop('Churn', axis=1)
y = encoded_df['Churn']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Development ---
# 1. Logistic Regression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

# 2. Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

# --- Model Evaluation ---
# 1. Logistic Regression Evaluation
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_auc = roc_auc_score(y_test, logistic_probabilities)
logistic_report = classification_report(y_test, logistic_predictions)
logistic_conf_matrix = confusion_matrix(y_test, logistic_predictions)

print("Logistic Regression Results:")
print(f"Accuracy: {logistic_accuracy}")
print(f"AUC: {logistic_auc}")
print("Classification Report:\n", logistic_report)
print("Confusion Matrix:\n", logistic_conf_matrix)

# 2. Random Forest Evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_auc = roc_auc_score(y_test, rf_probabilities)
rf_report = classification_report(y_test, rf_predictions)
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)

print("\nRandom Forest Results:")
print(f"Accuracy: {rf_accuracy}")
print(f"AUC: {rf_auc}")
print("Classification Report:\n", rf_report)
print("Confusion Matrix:\n", rf_conf_matrix)

# --- Visualization ---
# 1. Confusion Matrix Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(logistic_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 2. Feature Importance (Random Forest)
feature_importances = rf_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances (Random Forest)')
plt.show()
