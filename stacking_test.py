# coding=utf8


from sklearn import datasets
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from CPSC_config import Config
import CPSC_utils as utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from evaluate_12ECG_score import evaluate_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = Config()
config.MODEL_PATH = 'E:/challenge2020/CPSC_Scheme-master/model_t/'
config.MAN_FEATURE_PATH = 'E:/challenge2020/CPSC_Scheme-master/Man_features/'

records_name = np.array(os.listdir(config.DATA_PATH))
records_label = np.load(config.REVISED_LABEL) - 1
class_num = len(np.unique(records_label))

train_val_records, _, train_val_labels, test_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)

train_records, val_records, train_labels, val_labels = train_test_split(
    train_val_records, train_val_labels, test_size=0.2, random_state=config.RANDOM_STATE)

_, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
_, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

# 载入之前保存的网络输出概率以及人工特征 -------------------------------------------------------------------------------
pred_nnet_r = np.load(config.MODEL_PATH + 'pred_nnet_r.npy')
pred_nnet_v = np.load(config.MODEL_PATH + 'pred_nnet_v.npy')
pred_nnet_t = np.load(config.MODEL_PATH + 'pred_nnet_t.npy')

man_features_r = np.load(config.MAN_FEATURE_PATH + 'man_features_r.npy')
man_features_v = np.load(config.MAN_FEATURE_PATH + 'man_features_v.npy')
man_features_t = np.load(config.MAN_FEATURE_PATH + 'man_features_t.npy')

pred_r = np.concatenate((pred_nnet_r, man_features_r), axis=1)
pred_v = np.concatenate((pred_nnet_v, man_features_v), axis=1)
pred_train = np.concatenate((pred_r, pred_v), axis=0)
pred_t = np.concatenate((pred_nnet_t, man_features_t), axis=1)

lb_r = train_labels
lb_v = val_labels
lb_train=np.concatenate((lb_r, lb_v), axis=0)
lb_t = test_labels
for i in range(len(pred_train)):
    sample=pred_train[i]
    for j in range(len(sample)):
        if np.isnan(sample[j]):
            sample[j]=0
# print(np.isnan(pred_train).any())
# print(np.isnan(pred_t).any())

from sklearn.model_selection import KFold
def get_stacking(clf, x_train, y_train, x_test, n_folds=5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set



#我们这里使用5个分类算法，为了体现stacking的思想，就不加参数了
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

rf_model = RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy')
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)
et_model = ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy')
svc_model = SVC()


train_x, test_x, train_y, test_y = pred_train,pred_t,lb_train,lb_t

train_sets = []
test_sets = []
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
# for clf in [rf_model, gdbc_model, et_model]:
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

#使用决策树作为我们的次级分类器
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(meta_train, train_y)
df_predict = dt_model.predict(meta_test)

accuracy,f_measure,Fbeta_measure,Gbeta_measure=evaluate_score(lb_t, df_predict,class_num)

print('\nResult for test_set:--------------------\n')
print('accuracy:',accuracy)
print('f_measure:',f_measure)
print('Fbeta:',Fbeta_measure)
print('Gbeta:',Gbeta_measure)