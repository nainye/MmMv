import pandas as pd
import numpy as np


def make_train_test_dataset(HGG, LGG, test_size=0.33, random_state=404):
    HGG_test_num = int(HGG.shape[0] * test_size)
    HGG_train_num = HGG.shape[0] - HGG_test_num
    
    LGG_test_num = int(LGG.shape[0] * test_size)
    LGG_train_num = LGG.shape[0] - LGG_test_num
    
    np.random.seed(random_state)
    
    HGG_train_idx = np.random.choice(HGG.shape[0], HGG_train_num, replace=False)
    HGG_test_idx = np.setdiff1d(range(HGG.shape[0]), HGG_train_idx)
    
    LGG_train_idx = np.random.choice(LGG.shape[0], LGG_train_num, replace=False)
    LGG_test_idx = np.setdiff1d(range(LGG.shape[0]), LGG_train_idx)
    
    HGG_train = HGG[HGG_train_idx]
    HGG_train = np.concatenate([HGG_train[:,0], HGG_train[:,1], HGG_train[:,2], HGG_train[:,3], HGG_train[:,4], HGG_train[:,5], HGG_train[:,6]])
    
    HGG_test = HGG[HGG_test_idx]
    HGG_test = HGG_test[:,0]
    
    LGG_train = LGG[LGG_train_idx]
    LGG_train = np.concatenate([LGG_train[:,0], LGG_train[:,1], LGG_train[:,2], LGG_train[:,3], LGG_train[:,4], LGG_train[:,5], LGG_train[:,6]])
    
    LGG_test = LGG[LGG_test_idx]
    LGG_test = LGG_test[:,0]
    
    y_train = [1]*HGG_train.shape[0] + [0]*LGG_train.shape[0]
    y_test = [1]*HGG_test.shape[0] + [0]*LGG_test.shape[0]
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    X_train = np.concatenate([HGG_train, LGG_train])
    X_test = np.concatenate([HGG_test, LGG_test])
    
    shuffled = np.random.permutation(int(X_train.shape[0]))
    
    X_train = X_train[shuffled]
    y_train = y_train[shuffled]
    
    shuffled = np.random.permutation(int(X_test.shape[0]))
    
    X_test = X_test[shuffled]
    y_test = y_test[shuffled]
    
    return X_train, X_test, y_train, y_test
    

def main():
    dataset = np.load("dataset/BRATS2020_1(369,7,240,240,4,3).npy")
    metadata = pd.read_csv("dataset/name_mapping.csv")
    metadata = metadata[['Grade', 'BraTS_2020_subject_ID']]

    HGG = metadata[metadata['Grade'] == 'HGG']
    LGG = metadata[metadata['Grade'] == 'LGG']

    HGG_dataset = dataset[HGG.index]
    LGG_dataset = dataset[LGG.index]

    X_train, X_test, y_train, y_test = make_train_test_dataset(HGG_dataset, LGG_dataset, test_size=0.33, random_state=404)

    print("X_train's shape:", X_train.shape)
    print("X_test's shape:", X_test.shape)
    print("y_train's shape", y_train.shape)
    print("y_test's shape:", y_test.shape)
    
    if not os.path.isdir("dataset"):
        os.mkdir("dataset")
    np.save("dataset/BRATS2020_X_train_1", X_train)
    np.save("dataset/BRATS2020_X_test_1", X_test)
    np.save("dataset/BRATS2020_y_train_1", y_train)
    np.save("dataset/BRATS2020_y_test_1", y_test)

if __name__ == "__main__":
    main()