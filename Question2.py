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


# 80/20 Train-Test Split with Stratification
# stratify=y ensures the class balance is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Initialize and train the Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

#Accuracies of train and test
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

#What is entropy
# Entropy is the measure of in data, it helps decision trees make partitions
# Entropy values are between 0 and 1, with 1 meaning a completely balanced
# data set where all outcomes are evenly distributed and 0 meaning the
# data set is homogeneous, which is ideal for making partitions

# Based on the accuracy of the testing set, it looks like the model was
# trained well. If it was overfit it would not have performed as well
# on the test data