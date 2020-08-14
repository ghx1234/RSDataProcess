import numpy as np
import gdal
import rasterio
import keras
import sys
import os
from model import *
from tensorflow.python.keras import backend as K
import tensorflow as tf
from keras.callbacks import *
from keras.optimizers import SGD, Adam, Nadam
from sklearn.model_selection import train_test_split
from zengqiang import *
import random


def z_score(x):
    return (x - np.mean(x)) / np.std(x, ddof=1)

def get_full_path(name_txt, data_dir, label_dir, train=True):
    if train:
        fullpath, lablepath = [], []
        for name in open(name_txt):
            if name == '\n':
                continue
            fullpath.append(os.path.join(data_dir, name.strip('\n') ))
            lablepath.append(os.path.join(label_dir, name.strip('\n') ))
            # lablepath.append(os.path.join(label_dir, name.strip('\n')[:28] + '_label' + name.strip('\n')[28:]))
        X_train, X_test, y_train, y_test = train_test_split(fullpath, lablepath, test_size=test_size, random_state=0)
        return X_train, X_test, y_train, y_test
    else:
        fullpath, lablepath = [], []
        for name in open(name_txt):
            if name == '\n':
                continue
            fullpath.append(os.path.join(data_dir, name.strip('\n') ))
            lablepath.append(os.path.join(label_dir, name.strip('\n') ))
        # X_train, X_test, y_train, y_test = train_test_split(fullpath, lablepath, test_size=test_size, random_state=0)
        return fullpath, lablepath


def process_lines(full_path, label_path, augment=False):
    if augment:
        fanzhuan = random.randint(0, 3) - 1
        pingyiX, pingyiY = random.randint(-100, 100), random.randint(-100, 100)
        xuanzhuan, suofang = random.randint(0, 180) - 90, random.random() * 0.4 + 0.8
        liangdu = random.random() * 0.1 + 0.95

        panduan = random.randint(0,1)

    with rasterio.open(full_path) as ds:
        x = ds.read(
            out_shape=(DATA_BAND, IMAGE_SIZE_X, IMAGE_SIZE_Y),
            resampling=rasterio.enums.Resampling.bilinear
        )
        x = z_score(x)
        if augment:
            x = zengqiang(x)
            x.fanzhuan(fanzhuan)
            if panduan:
                x.xuanzhuansuofang(0, suofang)
            # x.liangdu(liangdu)
            else:
                x.pingyi(pingyiX, pingyiY)
            x = x.array
        x = np.transpose(x, (1, 2, 0))

    with rasterio.open(label_path) as ds:
        y = ds.read(
            out_shape=(1, IMAGE_SIZE_X, IMAGE_SIZE_Y),
            resampling=rasterio.enums.Resampling.bilinear
        )
        # y = np.where(y > 0, 1, y)
        if augment:
            y = zengqiang(y)
            y.fanzhuan(fanzhuan)
            if panduan:
                y.xuanzhuansuofang(0, suofang)
            # y.liangdu(liangdu)
            else:
                y.pingyi(pingyiX, pingyiY)
            y = y.array
        y = np.where(y > 0, 1, y)
        y = np.transpose(y, (1, 2, 0))

    return x, y


def generate_arrays_from_files(full_paths, label_paths):
    while 1:
        cnt = 0
        X =[]
        Y =[]
        for index in range(len(full_paths)):
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_lines(full_paths[index], label_paths[index], augment=True)
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt == BATCH_SIZE:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []


def val_Generator(full_paths, label_paths):
    while True:
        for index in range(len(full_paths)):
            x_train, y_train = process_lines(full_paths[index], label_paths[index])
            yield (np.expand_dims(x_train, 0), np.expand_dims(y_train, 0))


def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train():

    model = mwen(input_shape=(None, 512, 512, 4), class_num=CLASSES)
    # model = unet(input_shape=(None, 512, 512, 4), class_num=CLASSES)
    if os.path.exists(weights_path):
        model.load_weight(weights_path)
    optimizer = Adam(lr=learn_rate)
    model.compile(loss=softmax_sparse_crossentropy_ignoring_last_label, optimizer=optimizer,
                  metrics=[sparse_accuracy_ignoring_last_label])
    model.summary()

    ######callbacks
    checkpoint = ModelCheckpoint(filepath=os.path.join(out_path
                        , "weights-{epoch:02d}-{val_sparse_accuracy_ignoring_last_label:.2f}.hdf5"),
                save_weights_only=True, save_best_only=True, monitor='val_sparse_accuracy_ignoring_last_label')#.{epoch:d}

    tensorboard = TensorBoard(log_dir=os.path.join(out_path, 'logs'),
                              histogram_freq=10, write_graph=True)
    history = LossHistory()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    callbacks = [checkpoint, reduce_lr, history]

    x_train, x_test, y_train, y_test = get_full_path(name_txt, data_dir, label_dir)
    x_, y_ = get_full_path(val_txt, val_data_dir, val_label_dir, False)
    x_test, y_test = x_test + x_, y_test + y_

    steps_per_epoch = int(np.ceil(len(x_train) / float(BATCH_SIZE)))
    validation_steps = int(np.ceil(len(x_test) / float(BATCH_SIZE)))
    print(len(x_train), len(x_test))

    h = model.fit_generator(generate_arrays_from_files(x_train, y_train, ),
        epochs=epoch, shuffle=True, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
        validation_data=val_Generator(x_test, y_test),
                            validation_steps=validation_steps,
                            max_q_size=1000, verbose=1, nb_worker=1
                        )

    #     epochs=epoch, shuffle=False, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
    #     validation_data=val_Generator(val_name_txt, data_dir, label_dir, DATA_BAND, IMAGE_SIZE_X, IMAGE_SIZE_Y),
    #                         validation_steps=validation_steps,
    #                         max_q_size=1000, verbose=1, nb_worker=1
    #                     )
    with open(os.path.join(out_path, 'log.txt'), 'a', encoding='utf-8') as f:
        for i in history.losses:
            f.write(str(i))
            f.write('\n')

    model.save_weights(os.path.join(out_path, 'model.hdf5'))


if __name__ == '__main__':

    IMAGE_SIZE_X, IMAGE_SIZE_Y = 512, 512
    test_size = 0.1
    DATA_BAND = 4
    BATCH_SIZE = 4

    name_txt = '/home/zhoudengji/ghx/data/yangben/name.txt'
    data_dir = '/home/zhoudengji/ghx/data/yangben/img'
    label_dir = '/home/zhoudengji/ghx/data/yangben/sample'

    val_txt = '/home/zhoudengji/ghx/data/yangben/val.txt'
    val_data_dir = '/home/zhoudengji/ghx/data/yangben/val_img'
    val_label_dir = '/home/zhoudengji/ghx/data/yangben/val_label'

    out_path = '/home/zhoudengji/ghx/code/unet-master/womtfe'

    epoch = 50
    CLASSES = 2
    learn_rate = 0.001
    weights_path = 'None'

    train()


