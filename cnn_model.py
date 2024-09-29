# Model Training

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

import time
# IMPORT THE PREPROCESSED DATA
X = np.load('/content/drive/MyDrive/CNN PROJECT/ProcessedData (1)/image_array.npy')
y = np.load('/content/drive/MyDrive/CNN PROJECT/ProcessedData (1)/label_array (1).npy')

X.shape
# SPLIT DATA INTO TRAIN AND TEST
# 20% of data goes to testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=100)
# 20% of data goes to validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=len(y_test), random_state=100)
# TRAIN DATA USING 2D CONVOLUTIONAL NEURAL NETWORK
from sklearn.model_selection import KFold
import numpy as np
import time
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

# Parameters for layers
dense_layers = [1]
layer_sizes = [64]
conv_layers = [3]

# KFold setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_scores = []

# Loop through the configurations
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print("Starting model", NAME)

            fold_no = 1  # To keep track of fold number

            # Perform cross-validation
            for train_index, val_index in kf.split(X_train):
                print(f"Training fold {fold_no} for model {NAME}")

                # Create model
                model = Sequential()
                model.add(Conv2D(layer_size, (3, 3), input_shape=X_train.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_size, (3, 3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('relu'))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                # Compile model
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                # TensorBoard and ModelCheckpoint callbacks
                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
                filename = "/content/drive/MyDrive/model{}.keras".format(fold_no)
                checkpoint = ModelCheckpoint(filename, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

                # Get training and validation sets for the fold
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                # Train the model on the fold
                hist = model.fit(X_train_fold, y_train_fold,
                                 batch_size=32,
                                 epochs=6,
                                 validation_data=(X_val_fold, y_val_fold),
                                 callbacks=[checkpoint],
                                 verbose=1)

                # Evaluate the model on validation data
                val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
                print(f"Fold {fold_no} - Validation Accuracy: {val_acc}, Validation Loss: {val_loss}")
                all_scores.append(val_acc)

                fold_no += 1

# Calculate and print the average accuracy across all folds
average_score = np.mean(all_scores)
print(f"Average 10-Fold Cross-Validation Accuracy: {average_score:.4f}")
import matplotlib.pyplot as plt

# Plotting the accuracy graph
plt.figure(figsize=(5, 5))
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plotting the loss graph
plt.figure(figsize=(5, 5))
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
def GetFalseTruePositiveRate(y_true, y_prob, threshold):
    y_predict = np.fromiter([1 if x > threshold else 0 for x in y_prob], int)
    n_positives = y_true.sum()
    n_negatives = y_true.shape[0] - n_positives

    # Get counts for true positives, false positives, true negatives, and false negatives
    n_true_pos = np.sum((y_true == 1) & (y_predict == 1))
    n_false_pos = np.sum((y_true == 0) & (y_predict == 1))
    n_true_neg = np.sum((y_true == 0) & (y_predict == 0))
    n_false_neg = np.sum((y_true == 1) & (y_predict == 0))

    true_pos_rate = n_true_pos / n_positives if n_positives > 0 else 0
    false_pos_rate = n_false_pos / n_negatives if n_negatives > 0 else 0
    specificity = n_true_neg / (n_true_neg + n_false_pos) if (n_true_neg + n_false_pos) > 0 else 0

    return false_pos_rate, true_pos_rate, specificity


def MakeConfusionMatrix(y_true, y_prob, threshold):
    y_predict = np.fromiter([1 if x > threshold else 0 for x in y_prob], int)
    confusion_matrix = np.array([[0, 0], [0, 0]])

    for pred_value, true_value in zip(y_predict, y_true):
        if true_value == 1:
            if pred_value == 1:
                confusion_matrix[0, 0] += 1  # True Positive
            else:
                confusion_matrix[1, 0] += 1  # False Negative
        else:
            if pred_value == 1:
                confusion_matrix[0, 1] += 1  # False Positive
            else:
                confusion_matrix[1, 1] += 1  # True Negative

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    sns.heatmap(confusion_matrix, ax=ax, cmap='Blues', annot=True, fmt='g',
                xticklabels=['Infected', 'Uninfected'],
                yticklabels=['Infected', 'Uninfected'])
    ax.set_ylabel('Actual', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.title('Confusion Matrix', fontsize=24)
    plt.show()


# Predict the results
y_predict = model.predict(X_test).flatten()  # Flatten predictions to match y_test shape

# Define thresholds
thresholds = np.arange(0.01, 1.01, 0.01)
thresholds = np.append(np.array([0, 0.00001, 0.001]), thresholds)

# Calculate ROC AUC
roc_auc = np.array([GetFalseTruePositiveRate(y_test, y_predict, n) for n in thresholds])
roc_auc = np.sort(roc_auc, axis=0)
roc_auc_value = roc_auc_score(y_test, y_predict)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
false_pos_rate, true_pos_rate, specificity = GetFalseTruePositiveRate(y_test, y_predict, 0.5)

# Prepare the text for the plot
text = 'AUC-ROC score = {:.3f}'.format(roc_auc_value)
text += '\nAccuracy = {:.3f}'.format(accuracy * 100)
text += '\nLoss = {:.3f}'.format(loss)
text += '\nSpecificity = {:.3f}'.format(specificity)

# Plot ROC Curve
fig = plt.figure(figsize=(7, 7))
ax = fig.gca()
ax.set_title('Malaria AUC-ROC Curve', fontsize=28)
ax.set_ylabel('True Positive Rate', fontsize=20)
ax.set_xlabel('False Positive Rate', fontsize=20)
ax.plot(roc_auc[:, 0], roc_auc[:, 1])
ax.text(s=text, x=0.1, y=0.8, fontsize=20)
plt.show()

# Plot Confusion Matrix
MakeConfusionMatrix(y_test, y_predict, 0.5)

