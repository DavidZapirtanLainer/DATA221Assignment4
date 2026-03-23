from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
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


# Scaling applied for equal significance across all features
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

# Defining Early Stopping to prevent overfitting on our 900-row dataset
# monitor='val_loss': We watch the error on the 20% validation set, not already seen data
# patience=5: If the error doesn't improve for 5 straight epochs, we stop.
# restore_best_weights=True: This rolls the model back to its peak performance moment.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience= 10,
    restore_best_weights=True
)

# Training the model
# We use a batch_size of 16 to give the Adam optimizer more updates per epoch.
# We set epochs to 80, but Early Stopping will likely stop it much sooner (the 30-50 range)
# We use a validation split here to monitor performance on unseen data
# DURING training, so we can stop before the model overfits
neural_network_model.fit(
    X_train_scaled,
    y_train,
    epochs=80,
    batch_size=16, # The number of rows the model trains itself on before testing itself
    validation_split=0.2, #Like a practice quiz
    callbacks=[early_stop])



# 3. Report Accuracies
train_loss, train_acc = neural_network_model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_acc = neural_network_model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"TF Training Accuracy: {train_acc:.2f}")
print(f"TF Test Accuracy: {test_acc:.2f}")


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
