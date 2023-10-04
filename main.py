import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import os

from model.SP import SP
from model.CNNATTENTION import CNNATTENTION
from loss.EQLv2 import EQLv2
from utils import parse_data

# Set GPU device and disable eager execution
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.compat.v1.disable_eager_execution()

# Load and preprocess the training and testing data
train = pd.read_csv('C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\UNSW_NB15\\train_binary.csv')
test = pd.read_csv('C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\UNSW_NB15\\test_binary.csv')

X_train, y_train = parse_data(train, dataset_name='UNSW_NB15', classification_mode='binary')
X_test, y_test = parse_data(test, dataset_name='UNSW_NB15', classification_mode='binary')

# Reshape and preprocess the data
x_train_1 = np.reshape(X_train, (X_train.shape[0], 14, 14, 1)).astype('float32')
y_train_1 = y_train

x_test_2 = np.reshape(X_test, (X_test.shape[0], 14, 14, 1)).astype('float32')
y_test_2 = y_test

# Build the model
print(np.shape(x_train_1), np.shape(y_train_1), np.shape(x_test_2), np.shape(y_test_2))


# Define a function to build the model
def build_model(input_shape):
    cf = SP(classification_mode='binary')
    cf.build_graph(input_shape)
    eqloss = EQLv2()
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    cf.compile(optimizer=opt, loss=eqloss, metrics=['accuracy'])
    cf.summary()  # Corrected 'model' to 'cf'
    return cf


model = build_model((14, 14, 1))

# Define a learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=0,
    mode="auto",
    min_delta=0.01,
    cooldown=0,
    min_lr=0
)

# Train the model
model.fit(x_train_1, y_train_1, validation_data=(x_test_2, y_test_2), epochs=100, batch_size=16, callbacks=[reduce_lr])

# Evaluate the model
pred = model.predict(x_test_2)
pred = np.argmax(pred, axis=1)
y_eval = np.argmax(y_test_2, axis=1)
score = metrics.accuracy_score(y_eval, pred)
recall = metrics.recall_score(y_eval, pred)
cm = metrics.confusion_matrix(y_eval, pred)
fpr = cm[1][0] / (cm[1][0] + cm[1][1])
f1 = metrics.f1_score(y_eval, pred)

print(cm)
print("Validation score: {}".format(score))
print("DR: ", recall)
print("FPR: ", fpr)
print("F1: ", f1)
