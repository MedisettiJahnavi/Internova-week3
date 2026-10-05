# Student Performance Prediction
# Mini Project - Classification

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load the dataset
csv_path = Path(__file__).resolve().parent / "Book1.csv"
data = pd.read_csv(csv_path)

# Rename columns to match the expected feature names
column_mapping = {
    "Study hrs": "Study_Hours",
    "Attendence": "Attendance",
    "Previous Marks": "Previous_Marks",
    "Assignment Score": "Assignment_Score",
    "Project Score": "Project_Score",
}
data = data.rename(columns=column_mapping)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())

# Step 2: Clean the data
print("\nMissing values:")
print(data.isnull().sum())

# Remove rows containing missing values
data = data.dropna()

# Step 3: Select features and target
features = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score",
    "Project_Score",
]

X = data[features]
y = data["Result"].str.strip().str.title()

# Step 4: Convert Pass/Fail into numbers
y = y.map({
    "Fail": 0,
    "Pass": 1,
})

# Step 5: Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))

# Step 6: Create the Machine Learning model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=3,
)

# Step 7: Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Step 10: Display actual and predicted results
print("\nTest Results:")

for actual, predicted in zip(y_test, y_pred):
    actual_result = "PASS" if actual == 1 else "FAIL"
    predicted_result = "PASS" if predicted == 1 else "FAIL"

    print("Actual:", actual_result, "| Predicted:", predicted_result)

# Step 11: Test new students
new_students = pd.DataFrame({
    "Study_Hours": [6, 2, 8],
    "Attendance": [85, 60, 92],
    "Previous_Marks": [72, 45, 80],
    "Assignment_Score": [75, 50, 88],
    "Project_Score": [70, 45, 85],
})

predictions = model.predict(new_students)

print("\nNew Student Predictions:")

for i, prediction in enumerate(predictions):
    result = "PASS" if prediction == 1 else "FAIL"
    print(f"Student {i + 1}: {result}")
