from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Making dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
X = df.drop("target", axis = 1)
y = df['target']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

#Adding Max depth
clf = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth= 3)
clf.fit(X_train, y_train)

#Accuracies of train and test
# Train accuracy went down as the model was being more general, not memorizing data
# However this led to the test set accuracy being much better than before
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")


# Extract and display top 5 features
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Important Features (Constrained):")
print(importances.head(5))

# How controlling model complexity affects overfitting
# Controlling model complexity prevents overfitting by not allowing the model
# to learn of all the nuances of the data and make overly specific partitions


# How feature importance contributes to interpretability:
# Unlike black box models like Neural Networks Decision Trees allow us to see
# exactly which physical traits like worst concave points have the most impact
# on classification. Furthermore, by making a ranked list of importances
# clinicians or researchers  acn verify if the model is making decisions
# based on medically sound criteria rather than random correlations