import warnings
import numpy as np
from keras.layers import Conv1D, BatchNormalization, Activation, AveragePooling1D, Dense,LeakyReLU,ReLU
from keras.layers import Dropout, Concatenate, Flatten, Lambda
from keras import regularizers
from keras.layers import Reshape, LSTM, Bidirectional,GRU,CuDNNLSTM,CuDNNGRU,Softmax
from keras.layers import Add
from keras.layers import MaxPooling1D
# from keras.layers.core import Lambda
from biosppy.signals import ecg
from pyentrp import entropy as ent
import CPSC_utils as utils
from keras.layers import dot, concatenate
from keras import backend as K
from Config import Config
warnings.filterwarnings("ignore")


class Net2(object):
    """
        结合CNN和GRU（双向LSTM）的深度学习网络模型
    """
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    def __cnnblock(inp, ker=12, C=0.001, stridenum=1, initial='he_normal', poolnum=24):
        net = Conv1D(ker, 3,strides=stridenum, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = LeakyReLU(0.2)(net)
        net = Conv1D(ker, 3, strides=stridenum, padding='same', kernel_initializer=initial,
                     kernel_regularizer=regularizers.l2(C))(net)
        net = LeakyReLU(0.2)(net)
        # net = AveragePooling1D(poolnum, 2)(net)
        net=MaxPooling1D(3,3)(net)

        return net

    def __inception_block(inp, stride=1, activation='relu'):
        # k1, k2, k3, k4 = ince_filter
        # l1, l2, l3, l4 = ince_length
        inception = []

        x1 = Conv1D(12, 1, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(inp)
        x1 = BatchNormalization()(x1)
        x1 = Activation(activation)(x1)
        inception.append(x1)

        x2 = Conv1D(6, 1, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(inp)
        x2 = BatchNormalization()(x2)
        x2 = Activation(activation)(x2)
        x2 = Conv1D(12, 3, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation(activation)(x2)
        inception.append(x2)

        x3 = Conv1D(6, 1, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(inp)
        x3 = BatchNormalization()(x3)
        x3 = Activation(activation)(x3)
        x3 = Conv1D(12, 1, strides=stride, padding='same', kernel_regularizer=regularizers.l2(0.01))(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation(activation)(x3)
        inception.append(x3)

        x4 = MaxPooling1D(pool_size=3, strides=stride, padding='same')(inp)
        x4 = Conv1D(12, 1, strides=1, padding='same')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Activation(activation)(x4)
        inception.append(x4)
        v1 = Concatenate(axis=-1)(inception)

        return v1

    @staticmethod
    def __backbone(inp,C=0.001, initial='he_normal'):
        """
        # 用于信号片段特征学习的卷积层组合
        :param inp:  keras tensor, 单个信号切片输入
        :param C:   double, 正则化系数， 默认0.001
        :param initial:  str, 初始化方式， 默认he_normal
        :return: keras tensor, 单个信号切片经过卷积层后的输出
        """
        net = Net2.__cnnblock(inp, C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net, C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net, C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net, C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net,C=0.001, stridenum=1, initial='he_normal', poolnum=48)
        # net = Net2.__inception_block(inp)
        # net = Net2.__inception_block(net)
        # net = Net2.__inception_block(net)
        # net = Net2.__inception_block(net)
        # net = Net2.__inception_block(net)

        return net

    @staticmethod
    def nnet(inputs,num_classes,keep_prob=0.2):
        """
        # 适用于单导联的深度网络模型
        :param inputs: keras tensor, 切片并堆叠后的单导联信号.
        :param keep_prob: float, dropout-随机片段屏蔽概率.
        :param num_classes: int, 目标类别数.
        :return: keras tensor， 各类概率及全连接层前自动提取的特征.
        """
        bch = Net2.__backbone(inputs)
    # features = Net2.__backbone(inputs) //信号输入为单导联时
    #     features = Concatenate(axis=1)(branches)
        features = Dropout(keep_prob)(bch)
        # features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        # features = Bidirectional(CuDNNLSTM(12, return_sequences=False), merge_mode='concat')(features)
        features = Bidirectional(CuDNNGRU(12, return_sequences=True), merge_mode='concat')(features)

        # attention
        attention_pre = Dense(24, name='attention_vec')(features)  # [b_size,maxlen,64]
        attention_probs = Softmax()(attention_pre)  # [b_size,maxlen,64]
        attention_mul = Lambda(lambda x: x[0] * x[1])([attention_probs, features])
        # features = attention_3d_block1(features)
        features = BatchNormalization()(attention_mul)
        features = Flatten()(features)
        net = Dense(units=num_classes, activation='sigmoid')(features)
        return net, features




