# %% [markdown]
# # Classifying Iris Species using KNN

# %%
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# %%
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# %%
# Train the model
knn.fit(X_train, y_train)

# %%
# Predict on test set
y_pred = knn.predict(X_test)

# %%
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# %%
# Predict a single custom input
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example flower
prediction = knn.predict(sample)
print("Predicted class:", iris.target_names[prediction[0]])
