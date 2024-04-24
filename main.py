# Import necessary libraries


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset

data = pd.read_csv("water_potability.csv")

# Data exploration and visualization

# Distribution of individual features

plt.figure(figsize=(12, 10))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.histplot(data[col], bins=20, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Box plots to identify potential outliers

plt.figure(figsize=(12, 10))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x=data["Potability"], y=data[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Count plot for the target variable

plt.figure(figsize=(6, 4))
sns.countplot(x="Potability", data=data)
plt.title("Distribution of Potable and Non-Potable Water Samples")
plt.show()

# Split features and target variable

X = data.drop(columns=["Potability"])
y = data["Potability"]

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model

y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", train_accuracy)

y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print("Testing Accuracy:", test_accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred_test))

# Save the model for future use

joblib.dump(model, "water_potability_model.pkl")
