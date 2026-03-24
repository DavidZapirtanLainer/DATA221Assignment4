import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Q6: Convolutional Neural Network

# Load the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalize pixel values to [0, 1]
# Why: Normalization ensures that the input features have a similar scale,
# which helps the gradient descent optimizer converge faster.
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape to include channel dimension (28, 28, 1)
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

# Build the CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'), # Convolutional layer
    layers.MaxPooling2D((2, 2)),                  # MaxPooling layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')        # Output layer (10 classes)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=15, validation_split=0.1, verbose=1)

# Report Test Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")



# Q7: Error Analysis

# 1. Generate predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

# 2. Compute and display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix: Fashion MNIST")
plt.show()

# 3. Identify and visualize misclassified images
misclassified_idx = np.where(y_pred != y_test)[0]

plt.figure(figsize=(12, 4))
for i, idx in enumerate(misclassified_idx[:3]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()



# Q6 DISCUSSION:
# Why CNNs over Fully Connected (FC):
# CNNs are preferred because they preserve spatial hierarchy (the relationship between neighboring pixels).
# While FC  networks flatten images and lose local structure, CNNs use parameter sharing
# to detect the same pattern anywhere in the image.
# What the Convolution Layer learns:
# The filters in the convolution layer learn to detect low-level features such as edges,
# textures, and simple shapes (like the curve of a sleeve or the straight line of a trouser leg).



#Q7 DISCUSSION:
# Observed Pattern: A common pattern is the confusion between visually similar
#silhouettes, such as Shirt being misclassified as Coat or Pullover.
# Since these items share similar rectangular shapes and sleeve structures.
# Improvement Method: One realistic method is data augmentation (random
#horizontal flips or slight rotations). This forces the CNN to become
#invariant to minor orientation changes making it more robust.

