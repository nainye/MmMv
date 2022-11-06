import keras
import matplotlib.pyplot as plt

import skimage.transform

import keras
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import GlobalAveragePooling3D, MaxPooling3D
from keras.layers import Dense
from keras.layers import BatchNormalization, Activation
import numpy as np
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--view', type=int, default=0, help="inpute data\'s view axial(0)/coronal(1)/sagittal(2)")
args = parser.parse_args()

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
                  , loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.AUC(
    num_thresholds=200, curve='ROC',
    summation_method='interpolation', name=None, dtype=None,
    thresholds=None, multi_label=False, label_weights=None
), tf.keras.metrics.SensitivityAtSpecificity(0.5), tf.keras.metrics.SpecificityAtSensitivity(0.5)])
    
    return model

def main():
    X_train = np.load("dataset/BRATS2020_X_train_1.npy")
    y_train = np.load("dataset/BRATS2020_y_train_1.npy")

    X_test = np.load("dataset/BRATS2020_X_test_1.npy")
    y_test = np.load("dataset/BRATS2020_y_test_1.npy")

    X_train = X_train[:,:,:,:,args.view] # Axial(0) / Coronal(1) / Sagittal(2)
    X_test = X_test[:,:,:,:,args.view] # Axial(0) / Coronal(1) / Sagittal(2)

    X_train = X_train[:,:,:,:,np.newaxis]
    X_test = X_test[:,:,:,:,np.newaxis]

    initial_bias = np.log([76/293])
    output_bias = tf.keras.initializers.Constant(initial_bias)

    model = create_model()

    if args.view == 0:
        model = keras.models.load_model("3D_4Multimodal_Axial_epoch-17_valacc-0.90_valauc-0.95.h5")
    elif args.view == 1:
        model = keras.models.load_model("3D_4Multimodal_Coronal_epoch-02_valacc-0.93_valauc-0.97.h5")
    else:
        model = keras.models.load_model("3D_4Multimodal_Sagittal_epoch-05_valacc-0.86_valauc-0.91.h5")

    get_output = tf.keras.backend.function([model.layers[0].input],
                                       [model.layers[-3].output,  model.layers[-2].output, model.layers[-1].output])

    i = 40 # Sample num
    data = X_train[i]
    data = data[np.newaxis,:,:,:,:]

    [conv_outputs, gap_outputs, predictions] = get_output(data)
    class_weights = model.layers[-1].get_weights()[0]

    print("GT:", y_train[i])
    print("Prediction:", predictions)

    output = []
    for num, idx in enumerate(np.argmax(predictions,axis=1)):
        cam = tf.matmul(np.expand_dims(class_weights[:,idx],axis = 0),
                        np.transpose(np.reshape(conv_outputs[num],(30*30*4,256))))
        cam = tf.keras.backend.eval(cam)
        output.append(cam)
    result = np.reshape(output, (30,30,4))

    for i in range(4):
        result[:,:,i] = (result[:,:,i]-np.min(result[:,:,i]))/(np.max(result[:,:,i]) - np.min(result[:,:,i]))

    cam = np.uint8(255*result)

    # T1
    plt.imshow(np.uint8(255*data[0,:,:,0,0]), cmap='gray')
    plt.imshow(skimage.transform.resize(cam[:,:,0], (240,240)), alpha=0.5, cmap='jet')
    plt.colorbar()

    # T1CE
    plt.imshow(np.uint8(255*data[0,:,:,1,0]), cmap='gray')
    plt.imshow(skimage.transform.resize(cam[:,:,1], (240,240)), alpha=0.5, cmap='jet')
    plt.colorbar()

    # T2
    plt.imshow(np.uint8(255*data[0,:,:,2,0]), cmap='gray')
    plt.imshow(skimage.transform.resize(cam[:,:,2], (240,240)), alpha=0.5, cmap='jet')
    plt.colorbar()

    # T2 FLAIR
    plt.imshow(np.uint8(255*data[0,:,:,3,0]), cmap='gray')
    plt.imshow(skimage.transform.resize(cam[:,:,3], (240,240)), alpha=0.5, cmap='jet')
    plt.colorbar()

if __name__ == "__main__":
    main()