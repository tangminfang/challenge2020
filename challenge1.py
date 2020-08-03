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
from CPSC_model import Net3
from CPSC_config import Config
import CPSC_utils as utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as bk


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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
config = tf.ConfigProto()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()

DATA_PATH= '/home/zyhk/桌面/datanpy/'
REVISED_LABEL='/home/zyhk/桌面/CPSC_Scheme-master/recordlabel.npy'
records_name = np.array(os.listdir(DATA_PATH))
records_label = np.load(REVISED_LABEL) - 1
class_num = len(np.unique(records_label))

# 划分训练，验证与测试集 -----------------------------------------------------------------------------------------------
train_records, test_val_records, train_labels, test_val_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)
# del test_records, test_labels

test_records, val_records, test_labels, val_labels = train_test_split(
    test_val_records, test_val_labels, test_size=0.5, random_state=config.RANDOM_STATE)
del test_records, test_labels

# 过采样使训练和验证集样本分布平衡 -------------------------------------------------------------------------------------
train_records, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
val_records, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)

# 取出训练集和测试集病人对应导联信号，并进行切片和z-score标准化 --------------------------------------------------------
print('Fetching data ...-----------------\n')

# train_x = utils.Process_Leads(train_records,Path=DATA_PATH)
# # train_x = utils.Fetch_Pats_Lbs_sLead(train_records, Path=config.DATA_PATH,
# #                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
# #                                      seg_length=config.SEG_LENGTH)
train_y = to_categorical(train_labels, num_classes=class_num)
# val_x = utils.Process_Leads(val_records,Path=DATA_PATH)
# # val_x = utils.Fetch_Pats_Lbs_sLead(val_records, Path=DATA_PATH,
# #                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
# #                                      seg_length=config.SEG_LENGTH)
val_y = to_categorical(val_labels, num_classes=class_num)

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
batch_size = 32
epochs = 100
momentum = 0.9
keep_prob = 0.2
val_batchsize=32

bk.clear_session()
tf.reset_default_graph()
SEG_LENGTH = 36000
inputs = Input(shape=(SEG_LENGTH, 12))
net = Net3()
outputs, _ = net.nnet(inputs, num_classes=9, keep_prob=keep_prob)
model = Model(inputs=inputs, outputs=outputs)

opt = optimizers.Adam(lr=config.lr_schedule(epochs))
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

checkpoint = ModelCheckpoint(filepath=config.MODEL_PATH+model_name,
                             monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')
lr_scheduler = LearningRateScheduler(config.lr_schedule)
callback_lists = [checkpoint, lr_scheduler]
# model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=1,
#           validation_data=(val_x, val_y), callbacks=callback_lists)
model.fit_generator(minibatches(Path=DATA_PATH, inputsrecord=train_records, targets=train_y, batch_size=batch_size),
                    steps_per_epoch=len(train_records)//batch_size, epochs=epochs, callbacks=callback_lists
                    # validation_data=minibatches(Path=DATA_PATH, inputsrecord=val_records, targets=val_y, batch_size=val_batchsize),
                    # validation_steps=
                    )
del train_y

model = load_model(config.MODEL_PATH + model_name)

pred_vt = model.predict(val_x, batch_size=batch_size, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(val_y, axis=1)
del val_x, val_y

# 评估模型在验证集上的性能 ---------------------------------------------------------------------------------------------
Conf_Mat_val = confusion_matrix(true_v, pred_v)
print('Result-----------------------------\n')
print(Conf_Mat_val)
F1s_val = []
for j in range(class_num):
    f1t = 2 * Conf_Mat_val[j][j] / (np.sum(Conf_Mat_val[j, :]) + np.sum(Conf_Mat_val[:, j]))
    print('| F1-' + config.CLASS_NAME[j] + ':' + str(f1t) + ' |')
    F1s_val.append(f1t)

print('F1-mean: ' + str(np.mean(F1s_val)))
