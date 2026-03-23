from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 2. Train a Neural Network
# hidden_layer_sizes=(30,) creates one hidden layer with 30 neurons.
# activation='logistic' ensures the output unit uses a sigmoid function.
# max_iter=1000 provides enough epochs for the model to converge.
mlp = MLPClassifier(
    hidden_layer_sizes=(30,),
    activation='logistic',
    solver='adam',
    max_iter=1000,
    random_state=42
)


mlp.fit(X_train_scaled, y_train)

# 3. Report Accuracies
print(f"NN Training Accuracy: {mlp.score(X_train_scaled, y_train):.4f}")
print(f"NN Test Accuracy: {mlp.score(X_test_scaled, y_test):.4f}")

#Questions

# Why feature scaling is necessary for neural networks:
#Scaling is necessary in neural networks to ensure that all features contribute
# equally to the output, which is crucial for the model's performance
# Without scaling the model may be biased towards features with larger ranges
# leading to poor performance on features with smaller ranges

# 2. What an epoch represents:
# One epoch represents a single full iteration of the entire training dataset
# through the neural network. During an epoch the model sees every sample
# calculates the error/loss , and updates its  weights once per
# batch to improve future predictions.
