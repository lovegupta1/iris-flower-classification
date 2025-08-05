# wine_quality_knn.py

# 📦 Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 📂 Load Dataset
data = pd.read_csv("winequality-red.csv")  # Keep the file in same folder

# 🎯 Feature and Target
X = data.drop('quality', axis=1)
y = data['quality']

# 🔢 Preprocessing - Normalize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔀 Split the Data (Random split every time)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=None)

# 🤖 KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 🔍 Prediction and Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 📊 Output Results
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# 🔲 Confusion Matrix (using matplotlib)
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())  # Sorted quality values

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

# Axis labels
tick_marks = range(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Add numbers inside the boxes
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.show()