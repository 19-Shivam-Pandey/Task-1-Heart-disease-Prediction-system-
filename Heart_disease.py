import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV file
data = pd.read_csv('Heart_Disease_Prediction.xls')

# Get the column names
column_names = data.columns.tolist()

# Separate the features (X) and target variable (y)
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)

# Set the column names for X
X.columns = column_names[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Make predictions on new data
new_data = [[70, 1, 4, 130, 322, 0, 2, 109, 0, 2.4, 2, 3, 3]]
new_prediction = model.predict(new_data)

if new_prediction[0] == 1:
    print("The person is likely to have heart disease.")
else:
    print("The person is not likely to have heart disease.")