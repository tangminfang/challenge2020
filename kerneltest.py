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
from CPSC_model import Net2,Net3
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
# import xgboost as xgb

def dataload(Path, inputsrecord, targets,location,leadnum = 12, downsample = 2,buf_size=100):
    target_len = int(72000 / downsample)
    samplename = inputsrecord[location]
    samplelabel= targets[location]
    SEG_buf = np.zeros([1, target_len, leadnum], dtype=np.float32)
    SEGs = np.zeros([1, target_len, leadnum], dtype=np.float32)
    for i in range(len(samplename)):
        sig = np.load(Path + samplename[i])
        SEGt = np.float32(utils.sig_process(sig, target_length=target_len))
        SEG_buf = np.concatenate((SEG_buf, SEGt))
        del SEGt
        if SEG_buf.shape[0] >= buf_size:
            SEGs = np.concatenate((SEGs, SEG_buf[1:]))
            del SEG_buf
            SEG_buf = np.zeros([1, target_len, leadnum], dtype=np.float32)
    if SEG_buf.shape[0] > 1:
        SEGs = np.concatenate((SEGs, SEG_buf[1:]))
    del SEG_buf
    return SEGs[1:],samplelabel


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
seed = 7
numpy.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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

input_directory = '/home/better/桌面/tmf/Training_WFDB/'
# input_directory = '/home/zyhk/桌面/Training_WFDB'

def datafeatrecord(input_directory,records,downsample,buf_size=100,leadnum=12,featurenum=25):
    # input_files = []
    # for f in os.listdir(input_directory):
    #     if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
    #         input_files.append(f)


    classes=get_classes(input_directory,records)
    num_files = len(records)
    datalabel=[]
    target_len = int(72000 / downsample)
    SEG_buf = np.zeros([1, target_len, leadnum+1], dtype=np.float32)
    SEGs = np.zeros([1, target_len, leadnum+1], dtype=np.float32)
    # feat_buf=np.zeros([1,1,target_len], dtype=np.float32)
    featurezero=np.zeros([target_len,1])
    for i, f in enumerate(records):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        datalabel.append(getdata_class(header_data))
        features = np.asarray(get_12ECG_features(data, header_data))
        featurezero[0:featurenum,0]=features[0:featurenum]
        # feats_reshape = features.reshape(1, -1)
        feats_reshape = featurezero.reshape([1, featurezero.shape[0], featurezero.shape[1]])
        # feat_buf=np.concatenate((feat_buf,feats_reshape))

        SEGt = np.float32(utils.sig_process(data, target_length=target_len))
        SEGt = np.concatenate((SEGt, feats_reshape),axis=2)
        del data
        SEG_buf = np.concatenate((SEG_buf, SEGt))
        del SEGt
        if SEG_buf.shape[0] >= buf_size:
            SEGs = np.concatenate((SEGs, SEG_buf[1:]))
            del SEG_buf
            SEG_buf = np.zeros([1, target_len, leadnum+1], dtype=np.float32)
    if SEG_buf.shape[0] > 1:
        SEGs = np.concatenate((SEGs, SEG_buf[1:]))
    del SEG_buf
    datalabel = np.array(datalabel)
    return SEGs[1:], datalabel

def datazeros(inputdata,lead):
    SEG_buf = np.zeros([inputdata.shape[0], inputdata.shape[1], inputdata.shape[2]], dtype=np.float32)
    for i in range(inputdata.shape[0]):
        SEG_buf[i,:,lead]=inputdata[i,:,lead]
    return SEG_buf

def datarecord(input_directory):
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    num_files = len(input_files)
    datalabel=[]
    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        datalabel.append(getdata_class(header_data)-1)
    datalabel = np.array(datalabel)
    return np.array(input_files), datalabel

from keras import backend as K
import tensorflow as tf

def joint_opt_loss(soft_labels,preds, alpha=0.8, beta=0.4):
    # introduce prior prob distribution p

    p = K.ones(9) / 9

    prob = tf.nn.softmax(preds, dim=1)
    prob_avg = K.mean(prob, axis=0)

    # ignore constant
    # L_c = -K.mean(K.sum(soft_labels * tf.nn.log_softmax(preds, dim=1), axis=-1))
    L_c = K.categorical_crossentropy(soft_labels, preds)
    L_p = -K.sum(K.log(prob_avg) * p)
    L_e = -K.mean(K.sum(prob * tf.nn.log_softmax(preds, dim=1), axis=-1))

    loss = L_c + alpha * L_p + beta * L_e
    return loss

MODEL_PATH='/home/better/桌面/modelfile/'

recordnames,labeltotal=datarecord(input_directory)
train_val_records, test_records, train_val_labels, test_labels = train_test_split(
    recordnames, labeltotal, test_size=0.1, random_state=config.RANDOM_STATE)

train_records, val_records, train_labels, val_labels = train_test_split(
    train_val_records, train_val_labels, test_size=0.1, random_state=config.RANDOM_STATE)

del recordnames,labeltotal

# train_ecg, train_targets = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
# val_ecg, val_targets = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

train_data,train_targets=datafeatrecord(input_directory,train_records,4)
val_data, val_targets= datafeatrecord(input_directory,val_records,4)
# records,label,features,class_num=datarecord(input_directory,4)
train_data[np.isnan(train_data)]=0.0
train_data[np.isinf(train_data)]=0.0
train_targets=train_targets-1
val_data[np.isnan(val_data)]=0.0
val_data[np.isinf(val_data)]=0.0
val_targets=val_targets-1
leadnum=12
###############################################################################################

train_data1 = train_data[:,:,0:12]
val_data1 = val_data[:,:,0:12]
acc_oval_temp = []
f_oval_temp = []
Fbeta_oval_temp = []
Gbeta_oval_temp = []
auroc_temp = []
auprc_temp= []
kernum=0
num = 0
class_num=9
for kerv in range(5,30,5):

    # for train, val in kfold.split(train_val_data,label):
    model_name = 'net_train_' + str(num) + '.hdf5'
    train_y=to_categorical(train_targets, num_classes=class_num)
    inputs = Input(shape=(SEG_LENGTH, 12))
    net = Net2()
    outputs, _ = net.nnet(inputs,ker=kerv, num_classes=9, keep_prob=keep_prob)
    model = Model(inputs=inputs, outputs=outputs)

    opt = optimizers.Adam(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    # model.compile(optimizer=opt, loss=joint_opt_loss,
    #                             metrics=['categorical_accuracy'])

    checkpoint = ModelCheckpoint(filepath=MODEL_PATH + model_name,
                                 monitor='val_categorical_accuracy', mode='max',
                                 save_best_only='True')
    # lr_scheduler = LearningRateScheduler(config.lr_schedule)
    callback_lists = [checkpoint]
    test_y = to_categorical(val_targets, num_classes=class_num)
    model.fit(x=train_data1, y=train_y, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(val_data1, test_y), callbacks=callback_lists)
    loss, accuracy = model.evaluate(train_data1, train_y, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    print("loss: {:.4f}".format(loss))

    pred_y=model.predict(val_data1)
    pred_v = np.argmax(pred_y, axis=1)
    true_v = np.argmax(test_y, axis=1)
    m=0
    for i in range(len(pred_v)):
        if(pred_v[i]==true_v[i]):
            m=m+1
    acc_test=m/len(pred_v)
    acc_oval, f_oval, Fbeta_oval, Gbeta_oval,auroc, auprc = evaluate_score(test_y, pred_y, class_num)
    print('acc:', acc_oval,acc_test,end=' ')
    print('f:', f_oval,end=' ')
    print('Fbeta:', Fbeta_oval,end=' ')
    print('Gbeta:', Gbeta_oval,end=' ')
    print('auroc:', auroc, end=' ')
    print('auprc:', auprc, end=' ')
    acc_oval_temp.append(acc_oval)
    f_oval_temp.append(f_oval)
    Fbeta_oval_temp.append(Fbeta_oval)
    Gbeta_oval_temp.append(Gbeta_oval)
    auroc_temp.append(auroc)
    auprc_temp.append(auprc)


    # scores = model.evaluate(x=train_val_data[val], y=test_y, verbose=False)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    # cvscores.append(scores[1] * 100)
    #################################################################################
    num = num + 1

np.save("acc_oval_temp1.npy", acc_oval_temp)
np.save("f_oval_temp1.npy",f_oval_temp)
np.save("Fbeta_oval_temp1.npy", Fbeta_oval_temp)
np.save("Gbeta_oval_temp1.npy",Gbeta_oval_temp)
np.save("auroc_temp1.npy", auroc_temp)
np.save("auprc_temp1.npy",auprc_temp)

# del train_val_data
# splits_num=5
# # train_records, val_records, train_labels, val_labels,feature_train,feature_val = traintest_split(train_val_records, train_val_labels,
# #                                                                                 feat_train, 0.9, leadnum=12,target_len=SEG_LENGTH,buf_size=100)
# # train_val_records, test_records, train_val_labels, test_labels = train_test_split(
# #     records, label, test_size=0.1, random_state=config.RANDOM_STATE)
# train_records, val_records, train_labels, val_labels=train_test_split(train_val_records, train_val_labels, test_size=0.1, random_state=config.RANDOM_STATE)
# train_data = train_records[:, :, 0:12]
# val_data = val_records[:, :, 0:12]
# test_data = test_records[:, :, 0:12]
# featurenum = 25
# train_feature = train_records[:, 0:featurenum, 12]
# val_feature = val_records[:, 0:featurenum, 12]
# test_feature = test_records[:, 0:featurenum, 12]
# del train_records,val_records,test_records
#
# train_feature = train_feature.reshape([train_feature.shape[0],train_feature.shape[1]])
# val_feature = val_feature.reshape([val_feature.shape[0],val_feature.shape[1]])
# test_feature = test_feature.reshape([test_feature.shape[0],test_feature.shape[1]])
# for i in range(splits_num):
#     model_name = 'net_train_' + str(i) + '.hdf5'
#     model= load_model(MODEL_PATH + model_name)
#     ###########################model combine#########################################
#     # model predict
#     # layer_model = Model(inputs=model.input, outputs=model.layers[28].output)
#
#     # feature_layer_r = layer_model.predict(train_records, batch_size=64, verbose=1)
#     # feature_layer_v = layer_model.predict(val_records, batch_size=64, verbose=1)
#     # feature_layer_t = layer_model.predict(test_records, batch_size=64, verbose=1)
#     pred_train = model.predict(train_data, batch_size=64, verbose=1)
#     pred_val = model.predict(val_data, batch_size=64, verbose=1)
#     # train_yt = to_categorical(train_labels, num_classes=class_num)
#     # val_yt = to_categorical(val_labels, num_classes=class_num)
#
#     pred_test = model.predict(test_data, batch_size=64, verbose=1)
#     # test_yt = to_categorical(test_labels, num_classes=class_num)
#
#     # for j in range(leadnum):
#     #     traindata_lead = datazeros(train_data, j)
#     #     valdata_lead = datazeros(val_data, j)
#     #     testdata_lead= datazeros(test_data, j)
#     #     modellead_name = 'zeronet_train_lead' + str(j) + '_' + str(i) + '.hdf5'
#     #     modellead = load_model(MODEL_PATH + model_name)
#     #     predlead_train = modellead.predict(traindata_lead, batch_size=64, verbose=1)
#     #     predlead_val = modellead.predict(valdata_lead, batch_size=64, verbose=1)
#     #     predlead_test = modellead.predict(testdata_lead, batch_size=64, verbose=1)
#     #     if j == 0:
#     #         predlead_nnet_r = predlead_train[:, 1:]
#     #         predlead_nnet_v = predlead_val[:, 1:]
#     #         predlead_nnet_t = predlead_test[:, 1:]
#     #     else:
#     #         predlead_nnet_r = np.concatenate((predlead_nnet_r, predlead_train[:, 1:]), axis=1)
#     #         predlead_nnet_v = np.concatenate((predlead_nnet_v, predlead_val[:, 1:]), axis=1)
#     #         predlead_nnet_t = np.concatenate((predlead_nnet_t, predlead_test[:, 1:]), axis=1)
#     #
#     # pred_split_r = np.concatenate((pred_train[:, 1:], predlead_nnet_r), axis=1)
#     # pred_split_v = np.concatenate((pred_val[:, 1:], predlead_nnet_v), axis=1)
#     # pred_split_t = np.concatenate((pred_test[:, 1:], predlead_nnet_t), axis=1)
#
#     if i == 0:
#         # pred_nnet_r = pred_split_r
#         # pred_nnet_v = pred_split_v
#         # pred_nnet_t = pred_split_t
#         pred_nnet_r = pred_train[:, 1:]
#         pred_nnet_v = pred_val[:, 1:]
#         pred_nnet_t = pred_test[:, 1:]
#     else:
#         pred_nnet_r = np.concatenate((pred_nnet_r, pred_train[:, 1:]), axis=1)
#         pred_nnet_v = np.concatenate((pred_nnet_v, pred_val[:, 1:]), axis=1)
#         pred_nnet_t = np.concatenate((pred_nnet_t, pred_test[:, 1:]), axis=1)
#
#     # if i == 0:
#     #     pred_nnet_r = pred_train[:, 1:]
#     #     pred_nnet_v = pred_val[:, 1:]
#     #     pred_nnet_t = pred_test[:, 1:]
#     # else:
#     #     pred_nnet_r = np.concatenate((pred_nnet_r, pred_train[:, 1:]), axis=1)
#     #     pred_nnet_v = np.concatenate((pred_nnet_v, pred_val[:, 1:]), axis=1)
#     #     pred_nnet_t = np.concatenate((pred_nnet_t, pred_test[:, 1:]), axis=1)
#
#     ###############################################################################
#
# pred_r = np.concatenate((pred_nnet_r, train_feature), axis=1)
# pred_v = np.concatenate((pred_nnet_v, val_feature), axis=1)
# pred_t = np.concatenate((pred_nnet_t, test_feature), axis=1)
# pred_all=pred_r
# pred_all= np.concatenate((pred_all, pred_v), axis=0)
# pred_all= np.concatenate((pred_all, pred_t), axis=0)
#
# # pred_r = pred_nnet_r
# # pred_v = pred_nnet_v
# # pred_t = pred_nnet_t
#
# def standardize(X):
#     """特征标准化处理
#     Args:
#         X: 样本集
#     Returns:
#         标准后的样本集
#     """
#     m, n = X.shape
#     # 归一化每一个特征
#     meanValtemp=[]
#     stdtemp=[]
#     for j in range(n):
#         features = X[:,j]
#         meanVal = features.mean(axis=0)
#         std = features.std(axis=0)
#         meanValtemp.append(meanVal)
#         stdtemp.append(std)
#         if std != 0:
#             X[:, j] = (features-meanVal)/std
#         else:
#             X[:, j] = 0
#     meanValtemp=np.array(meanValtemp)
#     stdtemp=np.array(stdtemp)
#     np.save("meanVal.npy",meanValtemp)
#     np.save("stdVal.npy",stdtemp)
#     return X
#
# # pred_alls = standardize(pred_all)
# pred_rs = standardize(pred_r)
# pred_vs = standardize(pred_v)
# pred_ts = standardize(pred_t)
#
# lb_r = to_categorical(train_labels, num_classes=class_num)
# lb_v = to_categorical(val_labels, num_classes=class_num)
# lb_t = to_categorical(test_labels, num_classes=class_num)
#
# ######################################################################
# from sklearn.linear_model import LogisticRegression
# import joblib
#
# lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
# lr.fit(pred_rs, train_labels)
# # 评估在过采样验证集，原始验证集，以及测试集上的性能 -------------------------------------------------------------------
# pred = lr.predict_proba(pred_ts)
# # numpy.savetxt("result.txt", pred)
# # joblib.dump(filename='LR.model',value=lr)
# print('\nResult1 for test_set:--------------------\n')
# acc_oval, f_oval, Fbeta_oval, Gbeta_oval,auroc, auprc = evaluate_score(lb_t, pred, class_num)
# print('acc:', acc_oval,end=' ')
# print('f:', f_oval,end=' ')
# print('Fbeta:', Fbeta_oval,end=' ')
# print('Gbeta:', Gbeta_oval,end=' ')
# print('auroc:', auroc,end=' ')
# print('auprc:', auprc,end=' ')
#
# ############################################################################
# # from keras.models import Model
# # from keras.layers import Dense
# #
# # inputs = Input(shape=(545,))
# # hidden = Dense(256, activation='relu')(inputs)
# # hidden2 = Dense(128, activation='relu')(hidden)
# # hidden3 = Dense(64, activation='relu')(hidden2)
# # output = Dense(9, activation='softmax')(hidden3)
# #
# # modeldense = Model(inputs=inputs, outputs=output)
# # modeldense.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # modeldense.fit(pred_rs, lb_r, epochs=50, verbose=0)
# # modeldense.save('second_model.hdf5')
# # pred_t2 = modeldense.predict(pred_ts, batch_size=64, verbose=0)
# # print('\nResult2 for test_set:--------------------\n')
# # acc_oval, f_oval, Fbeta_oval, Gbeta_oval,auroc, auprc = evaluate_score(lb_t, pred_t2, class_num)
# # print('acc:', acc_oval,end=' ')
# # print('f:', f_oval,end=' ')
# # print('Fbeta:', Fbeta_oval,end=' ')
# # print('Gbeta:', Gbeta_oval,end=' ')
# # print('auroc:', auroc,end=' ')
# # print('auprc:', auprc,end=' ')
#
# # ##################################################################################
# from sklearn.svm import SVC
# svm = SVC(probability=True)
# svm.fit(pred_rs,train_labels)
# rfpred = svm.predict_proba(pred_ts)
# numpy.savetxt("result.txt", pred)
# joblib.dump(filename='svm1.model',value=svm)
# print('\nResult for test_set:--------------------\n')
# acc_oval, f_oval, Fbeta_oval, Gbeta_oval,auroc, auprc = evaluate_score(lb_t, rfpred, class_num)
# print('acc:', acc_oval,end=' ')
# print('f:', f_oval,end=' ')
# print('Fbeta:', Fbeta_oval,end=' ')
# print('Gbeta:', Gbeta_oval,end=' ')
# print('auroc:', auroc,end=' ')
# print('auprc:', auprc,end=' ')
