import librosa
import tensorflow as tf
import numpy as np
from Ctc import decode
from model_utils import ctc_lambda_func, norm_func

sr = 16000

train_input_val = tf.keras.layers.Input(name='the_input', shape=[None,num_features], dtype='float32')
seq_len = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')

gru_1 =  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(num_units, return_sequences=True,kernel_initializer='he_normal', name='gru1'))(train_input_val)
gru_2 =  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(num_units, return_sequences=True,kernel_initializer='he_normal', name='gru2'))(gru_1)
gru_3 =  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(num_units, return_sequences=True,kernel_initializer='he_normal', name='gru3'))(gru_2)
inner = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal', name='dense2')(gru_3)
y_pred = tf.keras.layers.Activation('softmax', name='softmax')(inner)
model2 = tf.keras.Model(inputs=[train_input_val,seq_len],outputs=y_pred)

model2.compile(loss={'softmax': lambda y_true, y_pred: y_pred}, optimizer='adam')
model2.load_weights('C:\\Users\\Niteya Shah\\Desktop\\CTC\\model_3.h5')

y,_ = librosa.load("D:\\work\\ML\\Dataset\\gu-in-Test\\Audios\\001960142.wav", sr = sr)
file = librosa.feature.mfcc(y, sr = sr , n_mfcc = 16)
train_input_data = norm_func(file.T)
model_prediction = model2.predict([[train_input_data],[train_input_data.shape[0]]])
dec = decode(model_prediction[0])
de_vectorize_string(dec[0], dict_corpora_func(True))
