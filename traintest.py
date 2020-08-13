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
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from CPSC_model import Net
from CPSC_config import Config
import CPSC_utils as utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as bk
from sklearn.model_selection import StratifiedKFold
import numpy
from driver import dataread,get_classes,getdata_class,load_challenge_data
from evaluate_12ECG_score import evaluate_score

def minibatches(Path, inputsrecord, targets, batch_size, leadnum = 12, downsample = 2):
    i=0
    target_len = int(72000 / downsample)
    while 1:  # 要无限循环
        labels = []
        indices = np.arange(len(inputsrecord))
        SEG_buf = np.zeros([1, target_len, leadnum],dtype=np.float32)

        for b in range(batch_size):
            if i == len(inputsrecord):
                i = 0
                np.random.shuffle(indices)
            samplename = inputsrecord[indices[i]]
            samplelabel= targets[i]
            labels.append(samplelabel)
            i += 1
            sig = np.load(Path + samplename)
            SEGt = np.float32(utils.sig_process(sig, target_length=target_len))
            SEG_buf = np.concatenate((SEG_buf, SEGt))
            del SEGt

        yield SEG_buf[1:], np.array(labels)

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
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
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

# train_x = utils.Process_Leads(train_records,Path=DATA_PATH)
# # train_x = utils.Fetch_Pats_Lbs_sLead(train_records, Path=config.DATA_PATH,
# #                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
# #                                      seg_length=config.SEG_LENGTH)

# val_x = utils.Process_Leads(val_records,Path=DATA_PATH)
# # val_x = utils.Fetch_Pats_Lbs_sLead(val_records, Path=DATA_PATH,
# #                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
# #                                      seg_length=config.SEG_LENGTH)

# for x, y in minibatches(Path=DATA_PATH, inputsrecord=train_records, targets=train_y, batch_size=6):
#     print('y的一个批次:', y)


model_name = 'net_1' + '.hdf5'

print('Scaling data ...-----------------\n')
# for j in range(train_x.shape[0]):
#     train_x[j, :] = scale(train_x[j, :], axis=0)
# for j in range(val_x.shape[0]):
#     val_x[j, :] = scale(val_x[j, :], axis=0)
#
# train_x = train_x.reshape([int(train_x.shape[0]), int(train_x.shape[1]),1])
# val_x = val_x.reshape([int(val_x.shape[0]), int(val_x.shape[1]),1])

# 设定训练参数，搭建模型进行训练 （仅根据验证集调参，以及保存性能最好的模型）-------------------------------------------
batch_size = 64
epochs = 50
momentum = 0.9
keep_prob = 0.2
val_batchsize=32

bk.clear_session()
tf.reset_default_graph()
SEG_LENGTH = 750
cvscores=[]

input_directory = '/usr/games/Training_WFDB'
# input_directory = '/home/zyhk/桌面/Training_WFDB'
def datarecord(input_directory,downsample,buf_size=100,leadnum=12):
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
    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        datalabel.append(getdata_class(header_data))
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
    return SEGs[1:], datalabel,len(classes)


def datarecord2(input_directory,target_lead,buf_size=100,segnum=24,seg_length=750, full_seg=True, stt=0):
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)


    classes=get_classes(input_directory,input_files)
    num_files = len(input_files)
    datalabel=[]
    SEG_buf = np.zeros([1, seg_length, segnum], dtype=np.float32)
    SEGs = np.zeros([1, seg_length, segnum], dtype=np.float32)
    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i + 1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        datalabel.append(getdata_class(header_data))
        datalead = data[target_lead,:]
        # SEGt = np.float32(utils.sig_process(data, target_length=target_len))
        SEGt = utils.Stack_Segs_generate2(datalead, seg_num=segnum, seg_length=seg_length, full_seg=full_seg, stt=stt)
        del data,datalead
        SEG_buf = np.concatenate((SEG_buf, SEGt))
        del SEGt
        if SEG_buf.shape[0] >= buf_size:
            SEGs = np.concatenate((SEGs, SEG_buf[1:]))
            del SEG_buf
            SEG_buf = np.zeros([1, seg_length, segnum], dtype=np.float32)
    if SEG_buf.shape[0] > 1:
        SEGs = np.concatenate((SEGs, SEG_buf[1:]))
    del SEG_buf
    datalabel = np.array(datalabel)
    return SEGs[1:], datalabel,len(classes)


records,label,class_num=datarecord2(input_directory,1)

label=label-1
for train, test in kfold.split(records,label):
    # train_x,train_y=dataload(Path=DATA_PATH, inputsrecord=records_name, targets=records_label,location=train)
    train_y=to_categorical(label[train], num_classes=class_num)
    inputs = Input(shape=(SEG_LENGTH, 24))
    net = Net()
    outputs, _ = net.nnet(inputs, num_classes=9, keep_prob=keep_prob)
    model = Model(inputs=inputs, outputs=outputs)

    opt = optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # checkpoint = ModelCheckpoint(filepath=config.MODEL_PATH+model_name,
    #                              monitor='val_categorical_accuracy', mode='max',
    #                              save_best_only='True')
    # lr_scheduler = LearningRateScheduler(config.lr_schedule)
    # callback_lists = [checkpoint, lr_scheduler]
    model.fit(x=records[train], y=train_y, batch_size=batch_size, epochs=epochs, verbose=1)
    loss, accuracy = model.evaluate(records[train], train_y, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    print("loss: {:.4f}".format(loss))
    test_y=to_categorical(label[test],num_classes=class_num)
    pred_y=model.predict(records[test])
    pred_v = np.argmax(pred_y, axis=1)
    true_v = np.argmax(test_y, axis=1)
    m=0
    for i in range(len(pred_v)):
        if(pred_v[i]==true_v[i]):
            m=m+1
    acc_test=m/len(pred_v)
    acc_oval, f_oval, Fbeta_oval, Gbeta_oval = evaluate_score(test_y, pred_y, class_num)
    print('acc:', acc_oval,acc_test,end=' ')
    print('f:', f_oval,end=' ')
    print('Fbeta:', Fbeta_oval,end=' ')
    print('Gbeta:', Gbeta_oval,end=' ')
    scores = model.evaluate(x=records[test], y=test_y, verbose=False)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    # # model.fit_generator(minibatches(Path=DATA_PATH, inputsrecord=train_records, targets=train_y, batch_size=batch_size),
    #                     steps_per_epoch=len(train_records)//batch_size, epochs=epochs, callbacks=callback_lists
    #                     # validation_data=minibatches(Path=DATA_PATH, inputsrecord=val_records, targets=val_y, batch_size=val_batchsize),
    #                     # validation_steps=
    #                     )
# del train_y

# model = load_model(config.MODEL_PATH + model_name)
#
# pred_vt = model.predict(val_x, batch_size=batch_size, verbose=1)
# pred_v = np.argmax(pred_vt, axis=1)
# true_v = np.argmax(val_y, axis=1)
# del val_x, val_y
#
# # 评估模型在验证集上的性能 ---------------------------------------------------------------------------------------------
# Conf_Mat_val = confusion_matrix(true_v, pred_v)
# print('Result-----------------------------\n')
# print(Conf_Mat_val)
# F1s_val = []
# for j in range(class_num):
#     f1t = 2 * Conf_Mat_val[j][j] / (np.sum(Conf_Mat_val[j, :]) + np.sum(Conf_Mat_val[:, j]))
#     print('| F1-' + config.CLASS_NAME[j] + ':' + str(f1t) + ' |')
#     F1s_val.append(f1t)
#
# print('F1-mean: ' + str(np.mean(F1s_val)))
