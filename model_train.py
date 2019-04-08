import time
import tensorflow as tf
import numpy as np
import json
import datetime as dt
from Corpora import de_vectorize_string,dict_corpora_func
from model_utils import ctc_lambda_func, norm_func

labels = np.array(np.load('train_labels.npy'))
train_input = np.array(np.load('train_inputs.npy'))
train_input = np.asarray([norm_func(i) for i in train_input])
seq_lens = np.array([[i.shape[0]] for i in train_input])
out_lens = np.array([[i.shape[0]] for i in labels])

num_features = 16
num_classes = 69

num_epochs = 14
num_hidden = 50
num_layers = 1
batch_size = 20
num_examples = train_input.shape[0]
num_batches_per_epoch = int(num_examples/batch_size)

num_units = 512

input_dataset = tf.data.Dataset.from_generator(lambda: train_input, tf.float32).padded_batch(batch_size, padded_shapes=([None, num_features]))
input_sequence_len =  tf.data.Dataset.from_tensor_slices(seq_lens).batch(batch_size)
output_targets = tf.data.Dataset.from_generator(lambda: labels, tf.int32).padded_batch(batch_size, padded_shapes = ([None]) , padding_values = -1)
output_sequence_len  = tf.data.Dataset.from_tensor_slices(out_lens).batch(batch_size)
dataset = tf.data.Dataset.zip((input_dataset, output_targets, input_sequence_len, output_sequence_len)).shuffle(1000).repeat(100)
dataset = dataset.prefetch(10)

iterator=tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
training_init_op=iterator.make_initializer(dataset)
next_element=iterator.get_next()

train_input_val = tf.keras.layers.Input(name='the_input', shape=[None,num_features], dtype='float32')
target_inputs = tf.keras.layers.Input(name='the_labels', shape=[1000], dtype='float32')
seq_len = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
out_len = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

gru_1 =  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(num_units, return_sequences=True,kernel_initializer='he_normal', name='gru1'))(train_input_val)
gru_2 =  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(num_units, return_sequences=True,kernel_initializer='he_normal', name='gru2'))(gru_1)
gru_3 =  tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(num_units, return_sequences=True,kernel_initializer='he_normal', name='gru3'))(gru_2)
inner = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal', name='dense2')(gru_3)
y_pred = tf.keras.layers.Activation('softmax', name='softmax')(inner)

loss_out = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, target_inputs, seq_len, out_len])
cp_callback = tf.keras.callbacks.ModelCheckpoint('C:\\Users\\Niteya Shah\\Desktop\\CTC\\Model_info\\model_best.h5',save_best_only=True,verbose=1)

model = tf.keras.Model(inputs=[train_input_val, target_inputs, seq_len, out_len],outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
model.fit(dataset.make_one_shot_iterator().get_next(),callbacks=[cp_callback], epochs = num_epochs, steps_per_epoch = num_batches_per_epoch)

tf.keras.models.save_model(model, 'C:\\Users\\Niteya Shah\\Desktop\\CTC\\model_3.h5')
