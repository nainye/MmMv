import pandas as pd
import numpy as np
import keras
import tensorflow as tf

random_state=404
np.random.seed(random_state)


def main():

    X_train_full = np.load("BRATS2020_X_train_1.npy")
    X_test_full = np.load("BRATS2020_X_test_1.npy")

    origin_idx = [i*7 for i in range(248)]
    X_train_full = X_train_full[origin_idx]

    def get_GAP(view=0):
        X_train = X_train_full[:,:,:,:,view]
        X_test = X_test_full[:,:,:,:,view]

        X_train = X_train[:,:,:,:,np.newaxis]
        X_test = X_test[:,:,:,:,np.newaxis]

        if view == 0:
            model = keras.models.load_model("3D_4Multimodal_Axial_epoch-17_valacc-0.90_valauc-0.95.h5")
        elif view == 1:
            model = keras.models.load_model("3D_4Multimodal_Coronal_epoch-02_valacc-0.93_valauc-0.97.h5")
        else:
            model = keras.models.load_model("3D_4Multimodal_Sagittal_epoch-05_valacc-0.86_valauc-0.91.h5")

        get_output = tf.keras.backend.function([model.layers[0].input],
                                                [model.layers[-2].output])
        X = X_train[1][np.newaxis]

        GAP_outputs_train = []
        for i in range(len(X_train)):
            X = X_train[i][np.newaxis]
            [gap] = get_output(X)
            GAP_outputs_train.append(gap)
        GAP_outputs_train = np.array(GAP_outputs_train)[:,0,:]

        GAP_outputs_test = []
        for i in range(len(X_test)):
            X = X_test[i][np.newaxis]
            [gap] = get_output(X)
            GAP_outputs_test.append(gap)
        GAP_outputs_test = np.array(GAP_outputs_test)[:,0,:]

        if view == 0:
            post = 'axial'
        elif view == 1:
            post = 'coronal'
        else:
            post = 'sagittal'
        
        TRAIN = pd.DataFrame(GAP_outputs_train)
        TRAIN.columns = [post+'_'+str(i+1) for i in range(256)]
        TEST = pd.DataFrame(GAP_outputs_test)
        TEST.columns = [post+'_'+str(i+1) for i in range(256)]

        return TRAIN, TEST

    TRAIN_axial, TEST_axial = get_GAP(0)
    TRAIN_coronal, TEST_coronal = get_GAP(1)
    TRAIN_sagittal, TEST_sagittal = get_GAP(2)

    TRAIN = pd.concat([TRAIN_axial, TRAIN_coronal, TRAIN_sagittal], axis=1)
    TEST = pd.concat([TEST_axial, TEST_coronal, TEST_sagittal], axis=1)
    if not os.path.isdir("dataset"):
        os.mkdir("dataset")
    TRAIN.to_csv("dataset/GAP_feature_train.csv", index=False)
    TEST.to_csv("dataset/GAP_feature_test.csv", index=False)


if __name__ == "__main__":
    main()