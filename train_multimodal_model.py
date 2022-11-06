import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import GlobalAveragePooling3D, MaxPooling3D
from keras.layers import Dense
from keras.layers import BatchNormalization, Activation
import random
from scipy import ndimage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--view', type=int, default=0, help="inpute data\'s view axial(0)/coronal(1)/sagittal(2)")
parser.add_argument('--GPU', type=int, default=1, help='using GPU(1) / using CPU(0)')
args = parser.parse_args()

if args.GPU == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def rotate_and_flip(volume):
    
    def scipy_rotae_and_flip(volume):
        angles = list(range(-30,30,1))
        vflips = [0,1]
        hflips = [0,1]

        angle = random.choice(angles)
        vflip = random.choice(vflips)
        hflip = random.choice(hflips)

        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        if vflip:
            volume = np.flip(volume,axis=0)
        if hflip:
            volume = np.flip(volume,axis=1)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume
    
    augmented_volume = tf.numpy_function(scipy_rotae_and_flip, [volume], tf.float64)
    return augmented_volume

def train_preprocessing(volume, label):
    volume = rotate_and_flip(volume)
    return volume, label

def validation_preprocessing(volume, label):
    return volume, label

def create_model():
    model = Sequential()
    model.add(Conv3D(64,(7,7,4),input_shape=[240,240,4,1],padding='same', 
                     kernel_initializer='random_normal', 
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(64,(7,7,4),padding='same',
                     kernel_initializer='random_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size = (2,2,1)))

    model.add(Conv3D(128,(5,5,4),padding='same',
                     kernel_initializer='random_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(128,(5,5,4),padding='same',
                     kernel_initializer='random_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size = (2,2,1)))

    model.add(Conv3D(256,(3,3,4),padding='same',
                     kernel_initializer='random_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(256,(3,3,4),padding='same',
                     kernel_initializer='random_normal',
                     bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size = (2,2,1)))
    
    model.add(GlobalAveragePooling3D())

    model.add(Dense(units = 1, activation = 'sigmoid',
                   kernel_initializer='random_normal',
                    bias_initializer=output_bias))
    
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                  , loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.SensitivityAtSpecificity(0.5), tf.keras.metrics.SpecificityAtSensitivity(0.5)])
    
    return model

def main():

    args = parser.parse_args()

    if args.view == 0:
        view = 'axial'
    elif args.view == 1:
        view = 'coronal'
    else:
        view = 'sagittal'

    X_train = np.load("dataset/BRATS2020_X_train_1.npy")
    y_train = np.load("dataset/BRATS2020_y_train_1.npy")

    X_test = np.load("dataset/BRATS2020_X_test_1.npy")
    y_test = np.load("dataset/BRATS2020_y_test_1.npy")

    X_train = X_train[:,:,:,:,args.view]
    X_test = X_test[:,:,:,:,args.view]

    X_train = X_train[:,:,:,:,np.newaxis]
    X_test = X_test[:,:,:,:,np.newaxis]

    train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test))


    BATCH_SIZE = 4
    train_dataset = (
        train_loader.shuffle(len(X_train))
        .map(train_preprocessing)
        .batch(BATCH_SIZE)
        .prefetch(2)
    )

    validation_dataset = (
        validation_loader.shuffle(len(X_test))
        .map(validation_preprocessing)
        .batch(BATCH_SIZE)
        .prefetch(2)
    )

    initial_bias = np.log([76/293])
    output_bias = tf.keras.initializers.Constant(initial_bias)

    model = create_model()

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3D_4Multimodal_"+view+"_epoch-{epoch:02d}_valacc-{val_accuracy:.2f}_valauc-{val_auc:.2f}.h5", monitor='val_auc'
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_auc", patience=15)

    history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=30,
                verbose=1,
                callbacks=[checkpoint_cb, early_stopping_cb]
    )


if __name__ == "__main__":
    main()