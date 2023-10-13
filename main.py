import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import os
import json
from pathlib import Path

from model.SP import SP
from model.CNNATTENTION import CNNATTENTION1D, CNNATTENTION2D
from LOGN.LOGL import create_logarithmic_layer
from loss.EQLv2 import EQLv2
from utils import parse_data
from configs.setting import setting
from dataset2image.main import deepinsight
from keras.models import Sequential, Model
from keras.layers import Input, Concatenate, Dense, Conv2D
from keras import regularizers

# Set GPU device and disable eager execution
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
# tf.compat.v1.disable_eager_execution()

i = 1
flag = True
SAVE_PATH_ = ''
TRAINED_MODEL_PATH_ = ''
CHECKPOINT_PATH_ = ''
config = {}
BASE_DIR = Path(__file__).resolve().parent
while flag:

    config, config_file = setting()
    TEMP_FILENAME = f"{config['DATASET_NAME']}-{config['CLASSIFICATION_MODE']}-{config['MODEL_NAME']}-{i}"
    TEMP_PATH = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

    if os.path.isdir(TEMP_PATH):
        i += 1
    else:
        flag = False

        os.mkdir(BASE_DIR.joinpath(f"session/{TEMP_FILENAME}"))
        SAVE_PATH_ = BASE_DIR.joinpath(f"session/{TEMP_FILENAME}")

        os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
        CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint.ckpt")

        with open(f'{SAVE_PATH_}/MODEL_CONFIG.json', 'w') as f:
            json.dump(config_file, f)

        print(f'MODEL SESSION: {SAVE_PATH_}')

# Load and preprocess the training and testing data
train = pd.read_csv(
    'C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\UNSW_NB15\\train_binary_withoutNorm_2neuron.csv')
test = pd.read_csv(
    'C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\UNSW_NB15\\test_binary_withoutNorm_2neuron.csv')

if config['DEEPINSIGHT']['use_deepinsight']:
    X_train, X_test = deepinsight(config['DEEPINSIGHT'], config)
    _, y_train = parse_data(train, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                            classification_mode=config['CLASSIFICATION_MODE'])
    _, y_test = parse_data(test, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                           classification_mode=config['CLASSIFICATION_MODE'])
else:
    X_train, y_train = parse_data(train, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                  classification_mode=config['CLASSIFICATION_MODE'])
    X_test, y_test = parse_data(test, dataset_name=config['DATASET_NAME'], mode=config['DATASET_TYPE'],
                                classification_mode=config['CLASSIFICATION_MODE'])

# Reshape and preprocess the data
# x_train_1 = np.reshape(X_train, (X_train.shape[0], 14, 14, 1)).astype('float32')
# X_train = np.reshape(X_train, (X_train.shape[0], 196, 1)).astype('float32')
#
# # x_test_2 = np.reshape(X_test, (X_test.shape[0], 14, 14, 1)).astype('float32')
# X_test = np.reshape(X_test, (X_test.shape[0], 196, 1)).astype('float32')
X_train = [X_train[:, i:i + 1].astype('float32') for i in range(X_train.shape[1])]
X_test = [X_test[:, i:i + 1].astype('float32') for i in range(X_test.shape[1])]
print(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))


# Define a function to build the model
def build_model(input_shape, optimizer):
    # input_layer = Input(shape=input_shape)

    # cf1 = SP(classification_mode='binary')
    # cf1.build_graph(input_shape)
    # out1 = cf1(input_layer)

    # cf2 = CNNATTENTION2D()
    # cf2.build_graph(input_shape)
    # out2 = cf2(input_layer)
    #
    # concatenated_output = Concatenate(axis=1)([out1, out2])
    # classification_output = Dense(2,
    #                               kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    #                               bias_regularizer=regularizers.L2(1e-4),
    #                               activity_regularizer=regularizers.L2(1e-5))(concatenated_output)

    # Define the model
    # cf = Model(inputs=input_layer, outputs=out1)
    mi_inputs, logarithmic_layer = create_logarithmic_layer(input_shape)
    output = CNNATTENTION1D()(logarithmic_layer)
    cf = Model(inputs=mi_inputs, outputs=output)
    # eqloss = EQLv2()
    criterion = tf.keras.losses.BinaryCrossentropy()
    cf.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])
    return cf


initial_learning_rate = config['LR']
final_learning_rate = config['MIN_LR']
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / config['EPOCHS'])
steps_per_epoch = int(len(X_train) / config['BATCH_SIZE'])
reduce_lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=steps_per_epoch,
    decay_rate=learning_rate_decay_factor,
    staircase=True)
opt = tf.keras.optimizers.legacy.Adam(reduce_lr)
model = build_model((196,), opt)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH_,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config['EPOCHS'], batch_size=config['BATCH_SIZE'],
          callbacks=[model_checkpoint])

# Evaluate the model
model = build_model((196, 1), opt)
model.load_weights(CHECKPOINT_PATH_).expect_partial()
pred = model.predict(X_test, verbose=2)
pred = np.argmax(pred, axis=1)
y_eval = np.argmax(X_test, axis=1)

result = dict()
result['score'] = metrics.accuracy_score(y_eval, pred)
result['recall'] = metrics.recall_score(y_eval, pred)
result['cm'] = metrics.confusion_matrix(y_eval, pred).tolist()
result['fpr'] = result['cm'][1][0] / (result['cm'][1][0] + result['cm'][1][1])
result['f1'] = metrics.f1_score(y_eval, pred)

print(result)
