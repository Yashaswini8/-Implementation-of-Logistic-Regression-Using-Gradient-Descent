# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1:Start
Step-2:Import the required packages and print the present data.
Step-3:Print the placement data and salary data.
Step-4:Find the null and duplicate values.
Step-5:Using logistic regression find the predicted values of accuracy , confusion matrices.
Step-6:Display the results.
Step-7:End

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Yashaswini S
RegisterNumber:  212224220123
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\admin\Downloads\Placement_Data.csv")

# 2. Preprocess
df.drop(columns=['sl_no', 'salary'], inplace=True)

# 3. Encode Categorical Variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 4. Define features and target
X = df.drop('status', axis=1).values
y = df['status'].values.reshape(-1, 1)

# 5. Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Logistic Regression (Manual)
# -----------------------------

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros((self.n, 1))
        self.bias = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Optional: print loss every 100 epochs
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
                print(f"Epoch {epoch} Loss: {loss:.4f}")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)

# Initialize and train model
model = LogisticRegressionGD(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

```

## Output:
<img width="1919" height="1199" alt="Screenshot 2025-09-27 194612" src="https://github.com/user-attachments/assets/2f1eeefd-6319-4e43-9822-9b42cf9c5539" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

