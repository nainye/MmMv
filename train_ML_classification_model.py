import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def main():
    X_train = pd.read_csv("dataset/GAP_feature_train.csv")
    y_train = np.load("dataset/BRATS2020_y_train_1.npy")

    X_test = pd.read_csv("dataset/GAP_feature_test.csv")
    y_test = np.load("dataset/BRATS2020_y_test_1.npy")

    origin_idx = [i*7 for i in range(248)]
    y_train = y_train[origin_idx]

    LR = LogisticRegression()
    SVM = SVC(probability=True)
    RF = RandomForestClassifier()

    LR.fit(X_train, y_train)
    SVM.fit(X_train, y_train)
    RF.fit(X_train, y_train)

    # train results
    LR_pred_train = LR.predict(X_train)
    SVM_pred_train = SVM.predict(X_train)
    RF_pred_train = RF.predict(X_train)

    LR_prob_train = LR.predict_proba(X_train)[:,1]
    SVM_prob_train = SVM.predict_proba(X_train)[:,1]
    RF_prob_train = RF.predict_proba(X_train)[:,1]

    LR_accuracy = accuracy_score(y_train, LR_pred_train)
    SVM_accuracy = accuracy_score(y_train, SVM_pred_train)
    RF_accuracy = accuracy_score(y_train, RF_pred_train)

    LR_sensitivity = recall_score(y_train, LR_pred_train)
    SVM_sensitivity = recall_score(y_train, SVM_pred_train)
    RF_sensitivity = recall_score(y_train, RF_pred_train)

    tn, fp, fn, tp = confusion_matrix(y_train, LR_pred_train).ravel()
    LR_specificity = tn/(tn+fp)
    tn, fp, fn, tp = confusion_matrix(y_train, SVM_pred_train).ravel()
    SVM_specificity = tn/(tn+fp)
    tn, fp, fn, tp = confusion_matrix(y_train, RF_pred_train).ravel()
    RF_specificity = tn/(tn+fp)

    LR_auc = roc_auc_score(y_train, LR_pred_train)
    SVM_auc = roc_auc_score(y_train, SVM_pred_train)
    RF_auc = roc_auc_score(y_train, RF_pred_train)

    LR_f1 = f1_score(y_train, LR_pred_train)
    SVM_f1 = f1_score(y_train, SVM_pred_train)
    RF_f1 = f1_score(y_train, RF_pred_train) 

    Train_Performance = [[LR_accuracy, LR_sensitivity, LR_specificity, LR_auc, LR_f1],
                        [SVM_accuracy, SVM_sensitivity, SVM_specificity, SVM_auc, SVM_f1],
                        [RF_accuracy, RF_sensitivity, RF_specificity, RF_auc, RF_f1]]
    Train_Performance = pd.DataFrame(Train_Performance)
    Train_Performance.index = ['Logistic', 'SVM', 'RF']
    Train_Performance.columns = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'F1 score']

    Train_Performance = Train_Performance.append(pd.DataFrame(Train_Performance.mean(), columns=['Average']).transpose())
    print("Train Performance", Train_Performance)

    # test results
    LR_pred_test = LR.predict(X_test)
    SVM_pred_test = SVM.predict(X_test)
    RF_pred_test = RF.predict(X_test)

    LR_prob_test = LR.predict_proba(X_test)[:,1]
    SVM_prob_test = SVM.predict_proba(X_test)[:,1]
    RF_prob_test = RF.predict_proba(X_test)[:,1]

    LR_accuracy_test = accuracy_score(y_test, LR_pred_test)
    SVM_accuracy_test = accuracy_score(y_test, SVM_pred_test)
    RF_accuracy_test = accuracy_score(y_test, RF_pred_test)

    LR_sensitivity_test = recall_score(y_test, LR_pred_test)
    SVM_sensitivity_test = recall_score(y_test, SVM_pred_test)
    RF_sensitivity_test = recall_score(y_test, RF_pred_test)

    tn, fp, fn, tp = confusion_matrix(y_test, LR_pred_test).ravel()
    LR_specificity_test = tn/(tn+fp)
    tn, fp, fn, tp = confusion_matrix(y_test, SVM_pred_test).ravel()
    SVM_specificity_test = tn/(tn+fp)
    tn, fp, fn, tp = confusion_matrix(y_test, RF_pred_test).ravel()
    RF_specificity_test = tn/(tn+fp)

    LR_auc_test = roc_auc_score(y_test, LR_prob_test)
    SVM_auc_test = roc_auc_score(y_test, SVM_prob_test)
    RF_auc_test = roc_auc_score(y_test, RF_prob_test)

    LR_f1_test = f1_score(y_test, LR_pred_test)
    SVM_f1_test = f1_score(y_test, SVM_pred_test)
    RF_f1_test = f1_score(y_test, RF_pred_test)

    Test_Performance = [[LR_accuracy_test, LR_sensitivity_test, LR_specificity_test, LR_auc_test, LR_f1_test],
                        [SVM_accuracy_test, SVM_sensitivity_test, SVM_specificity_test, SVM_auc_test, SVM_f1_test],
                        [RF_accuracy_test, RF_sensitivity_test, RF_specificity_test, RF_auc_test, RF_f1_test]]
    Test_Performance = pd.DataFrame(Test_Performance)
    Test_Performance.index = ['Logistic', 'SVM', 'RF']
    Test_Performance.columns = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'F1 score']

    Test_Performance = Test_Performance.append(pd.DataFrame(Test_Performance.mean(), columns=['Average']).transpose())
    print("Test Performance", Test_Performance)

if __name__ == "__main__":
    main()