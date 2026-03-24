from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Making dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
X = df.drop("target", axis = 1)
y = df['target']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

#Tree Model
clf = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth= 3)
clf.fit(X_train, y_train)

tree_predictions = clf.predict(X_test)


# Neural Network Model

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train a Neural Network

#Setting a random seed so the model returns same thing each time its run
tf.random.set_seed(1)


neural_network_model = Sequential()

#Create Input layer of 30 neurons, one for each feature
input_layer = InputLayer(shape=(30,))
neural_network_model.add(input_layer)


hidden_layer = Dense(24, activation= "relu")
neural_network_model.add(hidden_layer)

# Using sigmoid function to get output between 0 and 1
output_layer = Dense(1, activation='sigmoid')
neural_network_model.add(output_layer)

# Compile model and set loss function as binary_crossentropy for binary classification
neural_network_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience= 10,
    restore_best_weights=True
)

neural_network_model.fit(
    X_train_scaled,
    y_train,
    epochs=80,
    batch_size=16, # The number of rows the model trains itself on before testing itself
    validation_split=0.2, #Like a practice quiz
    callbacks=[early_stop])


nn_model_probabilities = neural_network_model.predict(X_test_scaled)
#
#
# # Converting to 1s and 0s
# # Sigmoid outputs are between 0-1, but 0s and 1s are needed for our accuracy
nn_predictions = (nn_model_probabilities >= 0.5).astype(int)


# Getting the confusion matrices
# Decision Tree Confusion Matrix
tree_cm = confusion_matrix(y_test, tree_predictions)
print("Decision Tree Confusion Matrix:", tree_cm)

# Neural Network Confusion Matrix
# Reshaping predictions to 1D array to match y_test
nn_cm = confusion_matrix(y_test, nn_predictions.flatten())
print("Neural Network Confusion Matrix:", nn_cm)

# --- Visualizing (Optional but helpful for analysis) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(tree_cm, display_labels=data.target_names).plot(ax=axes[0])
axes[0].set_title("Decision Tree")
ConfusionMatrixDisplay(nn_cm, display_labels=data.target_names).plot(ax=axes[1])
axes[1].set_title("Neural Network")
plt.show()

# Preference
# I would prefer the Neural network over the Decision tree since it has less false negatives
# We want our healthcare models to not miss any positives since it is a life or death
# situation


# Decision Tree:
# Advantage: High interpretability; we can see exactly which feature thresholds
#lead to a diagnosis, making it easy to explain to medical staff.
# Limitation: Prone to instability small changes in the training data can
# result in a completely different tree structure

# Neural Network:
# Advantage: Higher capacity for capturing complex, non-linear relationships
#between the 30 different features that a shallow tree might miss.
# Limitation: "Black box" nature; it is difficult to explain the specific
#reasoning behind an individual prediction to a patient or doctor