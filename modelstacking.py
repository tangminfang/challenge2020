from sklearn.model_selection import KFold
import numpy as np
import os

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

# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
from keras.layers import Reshape
from keras import optimizers
from keras.layers import Input
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from CPSC_model import Net2
from CPSC_config import Config
import CPSC_utils as utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as bk
from sklearn.model_selection import StratifiedKFold
import numpy
from driver import dataread,get_classes,getdata_class,load_challenge_data,get_12ECG_features
from evaluate_12ECG_score import evaluate_score
import xgboost as xgb


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# DATA_PATH= '/home/zyhk/桌面/datanpy/'
# REVISED_LABEL='/home/zyhk/桌面/CPSC_Scheme-master/recordlabel.npy'
# records_name = np.array(os.listdir(DATA_PATH))
# records_label = np.load(REVISED_LABEL) - 1
# class_num = len(np.unique(records_label))

# define 10-fold cross validation test harness
# seed = 7
# numpy.random.seed(seed)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# 取出训练集和测试集病人对应导联信号，并进行切片和z-score标准化 --------------------------------------------------------
print('Fetching data ...-----------------\n')

print('Scaling data ...-----------------\n')

# 设定训练参数，搭建模型进行训练 （仅根据验证集调参，以及保存性能最好的模型）-------------------------------------------
batch_size = 64
epochs = 50
momentum = 0.9
keep_prob = 0.2
val_batchsize=32

bk.clear_session()
tf.reset_default_graph()
SEG_LENGTH = 18000
cvscores=[]

input_directory = '/usr/games/Training_WFDB'
# input_directory = '/home/zyhk/桌面/Training_WFDB'
def datarecord(input_directory,downsample,buf_size=100,leadnum=12,featurenum=14):
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)


    classes=get_classes(input_directory,input_files)
    num_files = len(input_files)
    datalabel=[]
    target_len = int(72000 / downsample)
    SEG_buf = np.zeros([1, target_len, leadnum], dtype=np.float32)
    SEGs = np.zeros([1, target_len, leadnum], dtype=np.float32)
    feat_buf=np.zeros([1, featurenum], dtype=np.float32)
    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        datalabel.append(getdata_class(header_data))
        features = np.asarray(get_12ECG_features(data, header_data))
        feats_reshape = features.reshape(1, -1)
        feat_buf=np.concatenate((feat_buf,feats_reshape))

        SEGt = np.float32(utils.sig_process(data, target_length=target_len))
        del data
        SEG_buf = np.concatenate((SEG_buf, SEGt))
        del SEGt
        if SEG_buf.shape[0] >= buf_size:
            SEGs = np.concatenate((SEGs, SEG_buf[1:]))
            del SEG_buf
            SEG_buf = np.zeros([1, target_len, leadnum], dtype=np.float32)
    if SEG_buf.shape[0] > 1:
        SEGs = np.concatenate((SEGs, SEG_buf[1:]))
    del SEG_buf
    datalabel = np.array(datalabel)
    return SEGs[1:], datalabel,feat_buf[1:],len(classes)

def traintest_split(inputsrecord, targets, features, train_size,leadnum=12,target_len=SEG_LENGTH,featurenum=14, buf_size=100):
    splitnum=int(train_size*len(inputsrecord))
    labels = []
    indices = np.arange(len(inputsrecord))
    np.random.shuffle(indices)
    SEG_train = np.zeros([1, target_len, leadnum],dtype=np.float32)
    SEG_buf = np.zeros([1, target_len, leadnum], dtype=np.float32)
    feat_train=np.zeros([1, featurenum], dtype=np.float32)
    targets_train=[]
    SEG_test = np.zeros([1, target_len, leadnum], dtype=np.float32)
    SEG_buf1 = np.zeros([1, target_len, leadnum], dtype=np.float32)
    feat_test = np.zeros([1, featurenum], dtype=np.float32)
    targets_test = []
    for b in range(len(indices)):
        if b < splitnum:
            SEGtrain = inputsrecord[indices[b]]
            SEGtrain = SEGtrain.reshape([1, SEGtrain.shape[0], SEGtrain.shape[1]])
            SEG_buf = np.concatenate((SEG_buf, SEGtrain))
            targets_train.append(targets[indices[b]])
            feattrain=features[indices[b]]
            feattrain = feattrain.reshape([1, feattrain.shape[0]])
            feat_train=np.concatenate((feat_train,feattrain))
        else:
            SEGtest = inputsrecord[indices[b]]
            SEGtest = SEGtest.reshape([1, SEGtest.shape[0], SEGtest.shape[1]])
            SEG_buf1 = np.concatenate((SEG_buf1, SEGtest))
            targets_test.append(targets[indices[b]])
            feattest = features[indices[b]]
            feattest = feattest.reshape([1, feattest.shape[0]])
            feat_test = np.concatenate((feat_test, feattest))

        if SEG_buf.shape[0] >= buf_size:
            SEG_train = np.concatenate((SEG_train, SEG_buf[1:]))
            del SEG_buf
            SEG_buf = np.zeros([1, target_len, leadnum], dtype=np.float32)

        if SEG_buf1.shape[0] >= buf_size:
            SEG_test = np.concatenate((SEG_test, SEG_buf1[1:]))
            del SEG_buf1
            SEG_buf1 = np.zeros([1, target_len, leadnum], dtype=np.float32)

    if SEG_buf.shape[0] > 1:
        SEG_train = np.concatenate((SEG_train, SEG_buf[1:]))
    del SEG_buf

    if SEG_buf1.shape[0] > 1:
        SEG_test = np.concatenate((SEG_test, SEG_buf1[1:]))
    del SEG_buf1

    label_train = np.array(targets_train)
    label_test = np.array(targets_test)
    return SEG_train[1:], SEG_test[1:],label_train,label_test,feat_train[1:],feat_test[1:]


MODEL_PATH='/home/tmf'
records,label,features,class_num=datarecord(input_directory,4)
# train_val_records, test_records, train_val_labels, test_labels = train_test_split(
#     records, label, test_size=0.1, random_state=config.RANDOM_STATE)
records[np.isnan(records)]=0.0
records[np.isinf(records)]=0.0
label=label-1
num=0
train_val_records, test_records, train_val_labels, test_labels,feat_train,feat_test = traintest_split(records, label, features, 0.9,
                                                                                 leadnum=12,target_len=SEG_LENGTH,buf_size=100)
# del records,label
# for train, val in kfold.split(train_val_records,train_val_labels):
#     model_name = 'net_train_' + str(num) + '.hdf5'
#     train_y=to_categorical(label[train], num_classes=class_num)
#     inputs = Input(shape=(SEG_LENGTH, 12))
#     net = Net2()
#     outputs, _ = net.nnet(inputs, num_classes=9, keep_prob=keep_prob)
#     model = Model(inputs=inputs, outputs=outputs)
#
#     opt = optimizers.Adam(lr=0.01)
#     model.compile(optimizer=opt, loss='categorical_crossentropy',
#                   metrics=['categorical_accuracy'])
#
#     checkpoint = ModelCheckpoint(filepath=MODEL_PATH + model_name,
#                                  monitor='val_categorical_accuracy', mode='max',
#                                  save_best_only='True')
#     # lr_scheduler = LearningRateScheduler(config.lr_schedule)
#     callback_lists = [checkpoint]
#     test_y = to_categorical(label[val], num_classes=class_num)
#     model.fit(x=records[train], y=train_y, batch_size=batch_size, epochs=epochs, verbose=1,
#               validation_data=(records[val], test_y), callbacks=callback_lists)
#     loss, accuracy = model.evaluate(records[train], train_y, verbose=False)
#     print("Training Accuracy: {:.4f}".format(accuracy))
#     print("loss: {:.4f}".format(loss))
#
#     pred_y=model.predict(records[val])
#     pred_v = np.argmax(pred_y, axis=1)
#     true_v = np.argmax(test_y, axis=1)
#     m=0
#     for i in range(len(pred_v)):
#         if(pred_v[i]==true_v[i]):
#             m=m+1
#     acc_test=m/len(pred_v)
#     acc_oval, f_oval, Fbeta_oval, Gbeta_oval = evaluate_score(test_y, pred_y, class_num)
#     print('acc:', acc_oval,acc_test,end=' ')
#     print('f:', f_oval,end=' ')
#     print('Fbeta:', Fbeta_oval,end=' ')
#     print('Gbeta:', Gbeta_oval,end=' ')
#     scores = model.evaluate(x=records[val], y=test_y, verbose=False)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#     cvscores.append(scores[1] * 100)
#     num = num + 1
#
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
from sklearn.model_selection import KFold
def get_stacking(clf, x_train, y_train, x_test, n_folds=5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,9))
    second_level_test_set = np.zeros((test_num,9))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst = x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

splits_num=10
# train_records, val_records, train_labels, val_labels,feature_train,feature_val = traintest_split(train_val_records, train_val_labels,
#                                                                                 feat_train, 0.9, leadnum=12,target_len=SEG_LENGTH,buf_size=100)
for i in range(splits_num):
    model_name = 'net_train_' + str(i) + '.hdf5'
    model= load_model(MODEL_PATH + model_name)
    ###########################model combine#########################################
    # model predict
    layer_model = Model(inputs=model.input, outputs=model.layers[28].output)

    feature_layer_r = layer_model.predict(train_val_records, batch_size=64, verbose=1)
    feature_layer_t = layer_model.predict(test_records, batch_size=64, verbose=1)
    # pred_train = model.predict(train_records, batch_size=64, verbose=1)
    # pred_val = model.predict(val_records, batch_size=64, verbose=1)
    # train_yt = to_categorical(train_labels, num_classes=class_num)
    # val_yt = to_categorical(val_labels, num_classes=class_num)

    # pred_test = model.predict(test_records, batch_size=64, verbose=1)
    # test_yt = to_categorical(test_labels, num_classes=class_num)

    if i == 0:
        pred_nnet_r = feature_layer_r
        pred_nnet_t = feature_layer_t
    else:
        pred_nnet_r = np.concatenate((pred_nnet_r, feature_layer_r), axis=1)
        pred_nnet_t = np.concatenate((pred_nnet_t, feature_layer_t), axis=1)

    ###############################################################################

pred_r = np.concatenate((pred_nnet_r, feat_train), axis=1)
pred_t = np.concatenate((pred_nnet_t, feat_test), axis=1)


def standardize(X):
    """特征标准化处理
    Args:
        X: 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X


pred_rs = standardize(pred_r)
pred_ts = standardize(pred_t)
lb_r = to_categorical(train_val_labels, num_classes=class_num)
lb_t = to_categorical(test_labels, num_classes=class_num)
# lb_r = train_val_labels
# lb_t = test_labels

#我们这里使用5个分类算法，为了体现stacking的思想，就不加参数了
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

rf_model = RandomForestClassifier(n_estimators=5, n_jobs=-1, criterion='entropy')
adb_model = AdaBoostClassifier()
gdbc_model = GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=5)
et_model = ExtraTreesClassifier(n_estimators=5, n_jobs=-1, criterion='entropy')
svc_model = SVC()


train_sets = []
test_sets = []
for clf in [rf_model, adb_model, gdbc_model, et_model, svc_model]:
# for clf in [rf_model, gdbc_model, et_model]:
    train_set, test_set = get_stacking(clf, pred_rs, lb_r, pred_ts)
    train_sets.append(train_set)
    test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

#使用决策树作为我们的次级分类器
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(meta_train, lb_r)
df_predict = dt_model.predict(meta_test)

accuracy,f_measure,Fbeta_measure,Gbeta_measure, auroc, auprc=evaluate_score(lb_t, df_predict,class_num)

print('\nResult for test_set:--------------------\n')
print('accuracy:',accuracy)
print('f_measure:',f_measure)
print('Fbeta:',Fbeta_measure)
print('Gbeta:',Gbeta_measure)
print('auroc:', auroc, end=' ')
print('auprc:', auprc, end=' ')