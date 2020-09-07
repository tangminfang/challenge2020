# stacked generalization with neural net meta model on blobs dataset
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax
import os
import warnings
import numpy as np
from keras.models import Model, load_model
from keras.utils import to_categorical
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


# load models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        model_name = 'net_train_' + str(i) + '.hdf5'
        model = load_model(MODEL_PATH + model_name)
        # load model from file
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % model_name)
    return all_models


# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(20, activation='relu')(merge)
    output = Dense(9, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    # plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data
    inputy_enc = to_categorical(inputy)
    # fit model
    model.fit(X, inputy_enc, epochs=50, verbose=0)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


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
# load all models
n_members = 10
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on train dataset
fit_stacked_model(stacked_model, train_val_records, train_val_labels)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, test_records)
yhat = argmax(yhat, axis=1)
acc = accuracy_score(testy, yhat)
print('Stacked Test Accuracy: %.3f' % acc)