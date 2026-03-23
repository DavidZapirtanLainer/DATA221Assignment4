from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)


df['target'] = data.target

X = df.drop("target", axis = 1)
y = df['target']


#30 Columns with 569 rows
print(f"Shape of Feature Matrix: {X.shape}")

# One column with 569 Rows
print(f"Shape of Target {y.shape}")

# Here we print out the counts of 1s and 0s for the
# Targets to see if its balanced or not
target_counts = y.value_counts()
print(target_counts)

# Since There are 357 1s and 212 0s, this dataset is unbalanced

# Class balance is extremely important because if a dataset is extremely
# Unbalanced it will begin to favor the majority class and ignore the
# patterns of the minority class




