- 👋 Hi, I’m @RAMSWAROOP9521
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
RAMSWAROOP9521/RAMSWAROOP9521 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:gravity="center">

    <Button
        android:id="@+id/button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Click Me" />
</LinearLayout># Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

# Step 1: Load the dataset (Iris dataset)
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (species of flowers)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize a DecisionTreeClassifier model
# Here, we are 'changing' parameters such as max_depth to see how it impacts the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(12,8))
tree.plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

# Step 8: Change parameters (e.g., max_depth) and retrain
print("Training new model with a deeper tree (max_depth=5)...")
model_deeper = DecisionTreeClassifier(max_depth=5, random_state=42)
model_deeper.fit(X_train, y_train)
y_pred_deeper = model_deeper.predict(X_test)

# Step 9: Evaluate the new model
print("Accuracy with deeper tree:", metrics.accuracy_score(y_test, y_pred_deeper))

# Visualize the deeper decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(model_deeper, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

# Step 1: Load the dataset (Iris dataset)
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (species of flowers)

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize a DecisionTreeClassifier model
# Here, we are 'changing' parameters such as max_depth to see how it impacts the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Step 7: Visualize the Decision Tree
plt.figure(figsize=(12,8))
tree.plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

# Step 8: Change parameters (e.g., max_depth) and retrain
print("Training new model with a deeper tree (max_depth=5)...")
model_deeper = DecisionTreeClassifier(max_depth=5, random_state=42)
model_deeper.fit(X_train, y_train)
y_pred_deeper = model_deeper.predict(X_test)

# Step 9: Evaluate the new model
print("Accuracy with deeper tree:", metrics.accuracy_score(y_test, y_pred_deeper))

# Visualize the deeper decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(model_deeper, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
