# Model Training

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape

from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

#from tensorflow.keras.optimizers.legacy import Adam
from google.colab import drive
drive.mount('/content/drive')


def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


X = np.load('/content/drive/MyDrive/image_grayscale_array.npy')
y = np.load('/content/drive/MyDrive/label_grayscale_array.npy')


# 20% of data goes to testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=100)
# 20% of data goes to validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=len(y_test), random_state=100)

# print(X_train.shape)
# print(X_train.shape[1:])
# print(X_train.shape[1:], 1)

print(X_train.shape)
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2], 3)
#X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2], 3)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2], 3)

print(X_train.shape, X_val.shape, X_test.shape)
# print(y_train.shape)
print(y_train.shape)

# print(X_train[0:3])

model = Sequential()

# model.add(Reshape((150, 50)))

model.add(LSTM(units=50, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='softmax'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Assuming X and y are already defined and preprocessed
# X should be a 3D numpy array: (samples, timesteps, features)
# y should be a 1D or 2D array with binary labels

# Initialize K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1
metrics = []

# To store histories
history_per_fold = {}

for train_index, val_index in kf.split(X):
    print(f'\nTraining for Fold {fold_no} ...')

    # Split data
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Build the model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    opt = Adam(learning_rate=1e-3, decay=1e-5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=6,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Store history
    history_per_fold[fold_no] = history.history

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    metrics.append((val_loss, val_accuracy))

    print(f'Fold {fold_no} - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
    fold_no += 1

# Calculate average metrics across all folds
avg_loss = np.mean([m[0] for m in metrics])
avg_accuracy = np.mean([m[1] for m in metrics])
print(f'\nAverage Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}')

# Plotting Training and Validation Loss per Fold
plt.figure(figsize=(20, 10))

for fold in history_per_fold:
    plt.plot(history_per_fold[fold]['loss'], label=f'Fold {fold} Training Loss')
    plt.plot(history_per_fold[fold]['val_loss'], label=f'Fold {fold} Validation Loss')

plt.title('Training and Validation Loss per Fold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plotting Training and Validation Accuracy per Fold
plt.figure(figsize=(20, 10))

for fold in history_per_fold:
    plt.plot(history_per_fold[fold]['accuracy'], label=f'Fold {fold} Training Accuracy')
    plt.plot(history_per_fold[fold]['val_accuracy'], label=f'Fold {fold} Validation Accuracy')

plt.title('Training and Validation Accuracy per Fold')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Plot Average Metrics Across All Folds
# Aggregate loss and accuracy per epoch across all folds
epochs = range(1, 7)  # Since epochs=6

avg_train_loss = np.mean([history_per_fold[fold]['loss'] for fold in history_per_fold], axis=0)
avg_val_loss = np.mean([history_per_fold[fold]['val_loss'] for fold in history_per_fold], axis=0)
avg_train_acc = np.mean([history_per_fold[fold]['accuracy'] for fold in history_per_fold], axis=0)
avg_val_acc = np.mean([history_per_fold[fold]['val_accuracy'] for fold in history_per_fold], axis=0)

# Plot Average Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, avg_train_loss, label='Average Training Loss', marker='o')
plt.plot(epochs, avg_val_loss, label='Average Validation Loss', marker='o')
plt.title('Average Training and Validation Loss Across All Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Average Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, avg_train_acc, label='Average Training Accuracy', marker='o')
plt.plot(epochs, avg_val_acc, label='Average Validation Accuracy', marker='o')
plt.title('Average Training and Validation Accuracy Across All Folds')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Function to calculate precision, recall (sensitivity), specificity, and F1 score
def CalculateMetrics(y_true, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_predict)

    return specificity, precision, recall, f1

# Update the MakeConfusionMatrix function to return y_predict
def MakeConfusionMatrix(y_true, y_prob, threshold):
    confusion_mat = np.array([[0, 0], [0, 0]])
    for pred_value, true_value in zip(y_prob, y_true):
        if true_value == 1:
            if pred_value > threshold:
                confusion_mat[0, 0] += 1  # true positive
            else:
                confusion_mat[1, 0] += 1  # false negative
        else:
            if pred_value > threshold:
                confusion_mat[0, 1] += 1  # false positive
            else:
                confusion_mat[1, 1] += 1  # true negative

    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues',
                     xticklabels=['Infected', 'Uninfected'],
                     yticklabels=['Infected', 'Uninfected'])
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()

    return np.where(y_prob > threshold, 1, 0)  # Return predicted labels

# Predict and evaluate the model
y_predict_prob = model.predict(X_test).ravel()  # Ensure y_predict_prob is 1D
threshold = 0.5
y_predict = np.where(y_predict_prob > threshold, 1, 0)

# Calculate metrics
specificity, precision, recall, f1 = CalculateMetrics(y_test, y_predict)

# AUC-ROC evaluation
roc_auc_value = roc_auc_score(y_test, y_predict_prob)
loss, accuracy = model.evaluate(X_test, y_test)

text = (
    f'AUC-ROC score = {roc_auc_value:.3f}\n'
    f'Accuracy = {accuracy * 100:.3f}%\n'
    f'Loss = {loss:.3f}\n'
    f'Specificity = {specificity:.3f}\n'
    f'Precision = {precision:.3f}\n'
    f'Recall (Sensitivity) = {recall:.3f}\n'
    f'F1 Score = {f1:.3f}'
)

plt.figure(figsize=(7, 7))
ax = plt.gca()
ax.set_title('Malaria AUC-ROC Curve', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_xlabel('False Positive Rate', fontsize=12)

# Compute ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
ax.plot(fpr, tpr, label=f'AUC = {roc_auc_value:.3f}')
ax.legend(loc='lower right')

# Annotate metrics
ax.text(0.6, 0.2, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.show()

# Generate and display the confusion matrix
MakeConfusionMatrix(y_test, y_predict_prob, threshold)

# Model Evaluation

def GetFalseTruePositiveRate(y_true, y_prob, threshold):
    y_predict = np.fromiter([1 if x > threshold else 0 for x in y_prob], int)
    n_positives = y_true.sum()
    n_negatives = y_true.shape[0] - n_positives

    # get n true positives
    n_true_pos = 0
    n_false_pos = 0
    for pred_value, true_value in zip(y_predict, y_true):
        # true positive
        if true_value == 1 and pred_value == 1:
            n_true_pos += 1
        # false positive
        elif true_value == 0 and pred_value == 1:
            n_false_pos += 1
    true_pos_rate = n_true_pos / n_positives
    false_pos_rate = n_false_pos / n_negatives
    return false_pos_rate, true_pos_rate


def MakeConfusionMatrix(y_true, y_prob, threshold):
    confusion_matrix = np.array([[0, 0], [0, 0]])
    for pred_value, true_value in zip(y_prob, y_true):
        if true_value == 1:
            # true positive
            if pred_value > threshold:
                confusion_matrix[0, 0] += 1
            # false negative
            else:
                confusion_matrix[1, 0] += 1
        else:
            # false positive
            if pred_value > threshold:
                confusion_matrix[0, 1] += 1
            # true negative
            else:
                confusion_matrix[1, 1] += 1
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    sns.heatmap(confusion_matrix, ax=ax, cmap='Blues', annot=True, fmt='g',
                xticklabels=['Infected', 'Uninfected'],
                yticklabels=['Infected', 'Uninfected'])
    ax.set_ylabel('Actual', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.title('Confusion Matrix', fontsize=24)
    plt.show()


y_predict = model.predict(X_test)
thresholds = np.arange(0.01, 1.01, 0.01)
thresholds = np.append(np.array([0, 0.00001, 0.001]), thresholds)
roc_auc = np.array([GetFalseTruePositiveRate(y_test, y_predict, n) for n in thresholds])
roc_auc = np.sort(roc_auc, axis=0)
roc_auc_value = roc_auc_score(y_test, y_predict)
loss, accuracy = model.evaluate(X_test, y_test)
accuracy = accuracy
loss = loss
text = 'AUC-ROC score = {:.3f}'.format(roc_auc_value)
text += '\nAccuracy = {:.3f}'.format(accuracy * 100)
text += '\nLoss = {:.3f}'.format(loss)

fig = plt.figure(figsize=(7, 7))
ax = fig.gca()
ax.set_title('Malaria AUC-ROC Curve', fontsize=28)
ax.set_ylabel('True Positive Rate', fontsize=20)
ax.set_xlabel('False Positive Rate', fontsize=20)
ax.plot(roc_auc[:, 0], roc_auc[:, 1])
ax.text(s=text, x=0.1, y=0.8, fontsize=20)
plt.show()

MakeConfusionMatrix(y_test, y_predict, 0.5)

