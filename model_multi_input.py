"""
Dreem Data Challenge attempt by Fabien Roger, Alexandre Sajus
https://challengedata.ens.fr/participants/challenges/45/

Multi-input CNN attempt, one input branch with the signals, another with the Fourier Transform of some relevant signals
"""

# Imports
import h5py
import pandas as pd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy import signal

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv1D, MaxPool1D, ReLU, Input, Flatten, Reshape, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MSE
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score

import metric_dreem

# Amount of data to load (max: 4400)
n_data = 4400

# Loading data
PATH_TO_TRAINING_DATA = "X_train.h5"
PATH_TO_TRAINING_TARGET = "y_train.csv"
h5_file = h5py.File(PATH_TO_TRAINING_DATA)
# mask represents y, the results
mask = np.array(pd.read_csv(PATH_TO_TRAINING_TARGET))
file = h5py.File(PATH_TO_TRAINING_DATA, 'r')
data = file['data']

# Separating the different signals
N_signals = 8
x = data[:n_data, 2:]
x = x.reshape(n_data, N_signals, -1)
x = np.transpose(x, (0, 2, 1))  # shape = (n_data, 9000, 8)
mask = np.array(pd.read_csv(PATH_TO_TRAINING_TARGET))[
    :n_data, 1:]  # shape = (n_data, 90)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    x, mask, test_size=0.2, random_state=1)

# Calculating mean and variance of the signals for normalization
means = X_train.mean(axis=0).mean(axis=0)
stds = X_train.std(axis=0).mean(axis=0)


def normalize(X):   # Standardization
    return (X-means)/stds


X_train = normalize(X_train)
X_val = normalize(X_val)


def get_fft(X, channel):  # Get the Fourier transform for one of the 8 signals
    fs = 100
    d = X[:, :, channel]
    Sxx = signal.spectrogram(d, fs, nperseg=1024, noverlap=935)[2]
    Sxx = np.clip(Sxx, -10, 10)[:, :15, :]/5-1
    Sxx = np.transpose(Sxx, (0, 2, 1))
    return Sxx


# The channels where extracting the Fourier transform is relevant
channels = [0, 1, 2, 3, 6, 7]


def get_fft_inputs(X):  # Concatenate the FFT signals
    ffts = [get_fft(X, c) for c in channels]
    X_new = np.concatenate(ffts, axis=-1)
    return X_new


# Create FFT signals from the original signals
X_train_fft = get_fft_inputs(X_train)
X_val_fft = get_fft_inputs(X_val)


def f1(y_true, y_pred):  # A function to calculate the F1 score
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Multi_input network
# X_true is the input branch with the original signals
# X_fft is the input branch with the Fourier Transform of some relevant signals

# CNN 1D network for X_true
X_input_true = Input([9000, N_signals])

X_true = Conv1D(filters=20, kernel_size=20, padding='same')(X_input_true)
X_true = MaxPool1D(10)(X_true)
X_true = ReLU()(X_true)
X_true = Dropout(0.2)(X_true)
X_true = Conv1D(filters=40, kernel_size=20, padding='same')(X_true)
X_true = MaxPool1D(10)(X_true)
X_true = ReLU()(X_true)
X_true = Dropout(0.2)(X_true)

# CNN 1D network for X_fft
X_input_fft = Input([90, 90])
X_fft = Conv1D(filters=20, kernel_size=20, padding='same')(X_input_fft)
X_fft = ReLU()(X_fft)
X_fft = Dropout(0.2)(X_fft)

# Concatenate both inputs
combined = Concatenate()([X_true, X_fft])

# Output
z = Conv1D(filters=40, kernel_size=20, padding='same')(combined)
z = ReLU()(z)
z = Dropout(0.2)(z)
z = Conv1D(filters=1, kernel_size=10, padding='same',
           activation='sigmoid')(z)

#Compile and run
model = Model(inputs=[X_input_true, X_input_fft], outputs=z)
model.summary()

model.compile(optimizer=Adam(lr=0.0002),
              loss='binary_crossentropy',
                   metrics=['accuracy', f1])

# Callbacks to stop early in case of overfitting, save model at its best F1 score, reduce learning rate if F1 is on a plateau
my_callbacks = [EarlyStopping(monitor='val_f1', patience=10, verbose=0, mode='max'),
                ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True,
                                monitor='val_f1', mode='max'),
                ReduceLROnPlateau(monitor='val_f1', factor=0.1,
                                  patience=7, verbose=1, epsilon=1e-4, mode='max')
                ]

model.fit([X_train, X_train_fft], y_train, epochs=100,
          validation_data=([X_val, X_val_fft], y_val), callbacks=my_callbacks)

model.load_weights('.mdl_wts.hdf5')

"""
The next part does some optimizations on the predictions and calculates the best F1 score we can get
UNCOMMENTED
"""


def remove_smalls(y, min_size):
    n, m = y.shape
    for i in range(n):
        first1 = -1
        for j in range(m):
            if y[i, j] > 0.5 and first1 == -1:
                first1 = j
            if y[i, j] < 0.5 and first1 != -1:
                size = j - first1
                if size < min_size:
                    #print("remove !")
                    for k in range(first1, j):
                        y[i, k] = 0
                first1 = -1
        j = m
        size = j - first1
        if size < min_size:
            for k in range(first1, j):
                y[i, k] = 0


def remove_both_smalls(y, min_size_0, min_size_1):
    remove_smalls(y, min_size_0)
    y = 1-y
    remove_smalls(y, min_size_1)
    y = 1-y
    remove_smalls(y, min_size_0)
    return y


y_val_pred = model.predict([X_val, X_val_fft])[:, :, 0]
maxf = 0
max_p = None
l = [0, 3, 7, 12, 17, 22, 28]
# for min_size in [-1,0,1,2,3,4,5,7,10,12,14,16,18,20,22]:
for min_size_0 in l:
    for min_size_1 in l:
        for s in [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.8, 0.1, 0.2, 0.3, 0.5, 1]:
            y_pred_mask = np.zeros(y_val.shape)
            y_pred_mask[y_val_pred >= s] = 1
            yold = y_pred_mask.copy()
            y_pred_mask = remove_both_smalls(
                y_pred_mask, min_size_0, min_size_1)
            # print(sum(sum(abs(yold-y_pred_mask))))
            # print(min_size,s)
            try:
                f = metric_dreem.dreem_sleep_apnea_custom_metric(
                    y_val, y_pred_mask)
                # print(f)
                if f > maxf:
                    maxf = f
                    max_p = min_size_0, min_size_1, s, sum(
                        sum(abs(yold-y_pred_mask)))
            except:
                pass
print(maxf, max_p)
