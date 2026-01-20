# accuracy_gap.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# Step 1: Load CSV
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "C:/Users/HP/OneDrive/Documents/Desktop/AI- disease-prediction/dataset.csv")

data = pd.read_csv(file_path)
print("CSV loaded successfully!")
print(data.head())

# Step 2: Encode categorical columns

# Drop 'Name' (not useful as a feature)
X = data.drop(['Disease', 'Name'], axis=1)

# Target
y = data['Disease']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

print("Features:")
print(X.head())
print("Encoded Target:")
print(y)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Step 6: Plot accuracy gap
plt.figure(figsize=(6,4))
plt.bar(["Train Accuracy", "Test Accuracy"], [train_acc, test_acc], color=["blue", "orange"])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy Gap")
plt.text(0, train_acc + 0.02, f"{train_acc*100:.1f}%", ha='center', color='blue', fontsize=12)
plt.text(1, test_acc + 0.02, f"{test_acc*100:.1f}%", ha='center', color='orange', fontsize=12)
plt.show()
