# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:40:31 2018

@author: Winham

# CPSC_model.py:深度学习网络模型和人工HRV特征提取

"""

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



def attention_3d_block(hidden_states):
    # """
    # Many-to-one attention mechanism for Keras.
    # @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    # @return: 2D tensor with shape (batch_size, 128)
    # @author: felixhao28.
    # """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


from keras.layers import Multiply
from keras.layers.core import *


def attention_3d_block1(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS=74
    SINGLE_ATTENTION_VECTOR = False
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


class Net1(object):
    """
        结合CNN和RNN（双向LSTM）的深度学习网络模型
    """
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def _bn_relu(layer, config, dropout=0):
        layer = BatchNormalization()(layer)
        layer = Activation(config.conv_activation)(layer)

        if dropout > 0:
            layer = Dropout(config.conv_dropout)(layer)

        return layer

    @staticmethod
    def add_conv_weight(layer, filter_length, num_filters, config, subsample_length=1):
        layer = Conv1D(filters=num_filters, kernel_size=filter_length, strides=subsample_length, padding='same',
                       kernel_initializer=config.conv_init, kernel_regularizer=regularizers.l2(0.001))(layer)
        return layer

    @staticmethod
    def add_conv_layers(layer, config):
        for subsample_length in config.conv_subsample_lengths:
            layer = Net1.add_conv_weight(layer, config.conv_filter_length, config.conv_num_filters_start, config,
                                    subsample_length=subsample_length)
            layer = Net1._bn_relu(layer, config)
        return layer

    @staticmethod
    def resnet_block(layer, num_filters, subsample_length, block_index, config):
        def zeropad(x):
            y = K.zeros_like(x)
            return K.concatenate([x, y], axis=2)

        def zeropad_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 3
            shape[2] *= 2
            return tuple(shape)

        shortcut = MaxPooling1D(pool_size=subsample_length,padding='same')(layer)
        zero_pad = (block_index % config.conv_increase_channels_at) == 0 \
                   and block_index > 0
        if zero_pad is True:
            shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

        for i in range(config.conv_num_skip):
            if not (block_index == 0 and i == 0):
                layer = Net1._bn_relu(layer, config, dropout=config.conv_dropout if i > 0 else 0)
            layer = Net1.add_conv_weight(layer, config.conv_filter_length, num_filters, config,
                                    subsample_length if i == 0 else 1)
        layer = Add()([shortcut, layer])
        return layer

    @staticmethod
    def get_num_filters_at_index(index, num_start_filters, config):
        return 2 ** int(index / config.conv_increase_channels_at) \
               * num_start_filters

    @staticmethod
    def __add_resnet_layers(layer, config):
        layer = Net1.add_conv_weight(layer, config.conv_filter_length, config.conv_num_filters_start, config,
                                subsample_length=1)
        layer = Net1._bn_relu(layer, config)
        for index, subsample_length in enumerate(config.conv_subsample_lengths):
            num_filters = Net1.get_num_filters_at_index(index, config.conv_num_filters_start, config)
            layer = Net1.resnet_block(layer, num_filters, subsample_length, index, config)
        layer = Net1._bn_relu(layer, config)
        return layer

    @staticmethod
    def __backbone1(inp,config):
        layer = Net1.__add_resnet_layers(inp,config)
        # layer = GlobalAveragePooling1D()(layer)
        layer = AveragePooling1D(int(layer.shape[1]), int(layer.shape[1]))(layer)
        return layer

    @staticmethod
    def nnet(inputs, keep_prob, num_classes):
        """
        # 适用于单导联的深度网络模型
        :param inputs: keras tensor, 切片并堆叠后的单导联信号.
        :param keep_prob: float, dropout-随机片段屏蔽概率.
        :param num_classes: int, 目标类别数.
        :return: keras tensor， 各类概率及全连接层前自动提取的特征.
        """
        branches = []
        config=Config()
        for i in range(int(inputs.shape[-1])):
            ld = Lambda(Net1.__slice, output_shape=(int(inputs.shape[1]), 1), arguments={'index': i})(inputs)
            ld = Reshape((int(inputs.shape[1]), 1))(ld)
            bch = Net1.__backbone1(ld,config)
            branches.append(bch)
        features = Concatenate(axis=1)(branches)
        features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        features = Bidirectional(CuDNNLSTM(1, return_sequences=True), merge_mode='concat')(features)
        # features = attention_3d_block(features)
        features = Flatten()(features)
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features

class Net2(object):
    """
        结合CNN和GRU（双向LSTM）的深度学习网络模型
    """
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    def __cnnblock(inp, ker=3, C=0.001, stridenum=1, initial='he_normal', poolnum=24):
        net = Conv1D(12, ker,strides=stridenum, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = LeakyReLU(0.2)(net)
        net = Conv1D(12, ker, strides=stridenum, padding='same', kernel_initializer=initial,
                     kernel_regularizer=regularizers.l2(C))(net)
        net = LeakyReLU(0.2)(net)
        # net = AveragePooling1D(poolnum, 2)(net)
        net=MaxPooling1D(3,3)(net)

        return net

    @staticmethod
    def __backbone(inp,ker,C=0.001, initial='he_normal'):
        """
        # 用于信号片段特征学习的卷积层组合
        :param inp:  keras tensor, 单个信号切片输入
        :param C:   double, 正则化系数， 默认0.001
        :param initial:  str, 初始化方式， 默认he_normal
        :return: keras tensor, 单个信号切片经过卷积层后的输出
        """
        net = Net2.__cnnblock(inp,ker=ker, C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net, ker=ker,C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net,ker=ker, C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net,ker=ker, C=0.001, stridenum=1, initial='he_normal', poolnum=24)
        net = Net2.__cnnblock(net,ker=ker, C=0.001, stridenum=1, initial='he_normal', poolnum=48)

        return net

    @staticmethod
    def nnet(inputs,ker,num_classes,keep_prob=0.2):
        """
        # 适用于单导联的深度网络模型
        :param inputs: keras tensor, 切片并堆叠后的单导联信号.
        :param keep_prob: float, dropout-随机片段屏蔽概率.
        :param num_classes: int, 目标类别数.
        :return: keras tensor， 各类概率及全连接层前自动提取的特征.
        """
        bch = Net2.__backbone(inputs,ker)
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
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features

class Net3(object):
    """
        结合CNN和GRU（双向LSTM）的深度学习网络模型
    """
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def __cnnblock1(inp,kernelnum, C=0.001, stridenum=1, initial='he_normal'):
        net = Conv1D(kernelnum, 3,strides=stridenum, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        # net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Conv1D(kernelnum, 3, strides=stridenum, padding='same', kernel_initializer=initial,
                     kernel_regularizer=regularizers.l2(C))(net)
        # net = BatchNormalization()(net)
        net = Activation('relu')(net)
        # net = AveragePooling1D(poolnum, 2)(net)
        net=MaxPooling1D(3,3)(net)

        return net

    @staticmethod
    def __cnnblock2(inp, kernelnum, C=0.001, stridenum=1, initial='he_normal'):
        net = Conv1D(kernelnum, 3,strides=stridenum, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        # net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Conv1D(kernelnum, 3, strides=stridenum, padding='same', kernel_initializer=initial,
                     kernel_regularizer=regularizers.l2(C))(net)
        # net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Conv1D(kernelnum, 3, strides=stridenum, padding='same', kernel_initializer=initial,
                     kernel_regularizer=regularizers.l2(C))(net)
        # net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = MaxPooling1D(3,3)(net)

        return net

    @staticmethod
    def __backbone(inp, C=0.001, initial='he_normal'):
        """
        # 用于信号片段特征学习的卷积层组合
        :param inp:  keras tensor, 单个信号切片输入
        :param C:   double, 正则化系数， 默认0.001
        :param initial:  str, 初始化方式， 默认he_normal
        :return: keras tensor, 单个信号切片经过卷积层后的输出
        """
        net = Net3.__cnnblock1(inp, kernelnum=64, C=0.001, stridenum=1, initial='he_normal')
        net = Net3.__cnnblock1(net, kernelnum=128, C=0.001, stridenum=1, initial='he_normal')
        net = Net3.__cnnblock2(net, kernelnum=256, C=0.001, stridenum=1, initial='he_normal')
        net = Net3.__cnnblock2(net, kernelnum=512, C=0.001, stridenum=1, initial='he_normal')
        net = Net3.__cnnblock2(net, kernelnum=512, C=0.001, stridenum=1, initial='he_normal')

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

        features = Net3.__backbone(inputs)
        features = Dropout(keep_prob)(features)
        # features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        features = Bidirectional(GRU(32, return_sequences=False), merge_mode='concat')(features)
        # features = Bidirectional(CuDNNGRU(32, return_sequences=True), merge_mode='concat')(features)
        # features = attention_3d_block(features)
        features = BatchNormalization()(features)
        # features = Flatten()(features)
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features


class Net(object):
    """
        结合CNN和RNN（双向LSTM）的深度学习网络模型
    """
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def __backbone(inp, C=0.001, initial='he_normal'):
        """
        # 用于信号片段特征学习的卷积层组合
        :param inp:  keras tensor, 单个信号切片输入
        :param C:   double, 正则化系数， 默认0.001
        :param initial:  str, 初始化方式， 默认he_normal
        :return: keras tensor, 单个信号切片经过卷积层后的输出
        """
        net = Conv1D(4, 31, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 11, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 7, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(16, 5, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(int(net.shape[1]), int(net.shape[1]))(net)

        return net

    @staticmethod
    def nnet(inputs, keep_prob, num_classes):
        """
        # 适用于单导联的深度网络模型
        :param inputs: keras tensor, 切片并堆叠后的单导联信号.
        :param keep_prob: float, dropout-随机片段屏蔽概率.
        :param num_classes: int, 目标类别数.
        :return: keras tensor， 各类概率及全连接层前自动提取的特征.
        """
        branches = []
        for i in range(int(inputs.shape[-1])):
            ld = Lambda(Net.__slice, output_shape=(int(inputs.shape[1]), 1), arguments={'index': i})(inputs)
            ld = Reshape((int(inputs.shape[1]), 1))(ld)
            bch = Net.__backbone(ld)
            branches.append(bch)
        features = Concatenate(axis=1)(branches)
        features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        features = Bidirectional(CuDNNLSTM(1, return_sequences=True), merge_mode='concat')(features)
        # features = attention_3d_block(features)
        features = Flatten()(features)
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features


class ManFeat_HRV(object):
    """
        针对一条记录的HRV特征提取， 以II导联为基准
    """
    FEAT_DIMENSION = 15

    def __init__(self, sig, fs=250.0):
        assert len(sig.shape) == 1, 'The signal must be 1-dimension.'
        assert sig.shape[0] >= fs * 6, 'The signal must >= 6 seconds.'
        self.sig = utils.WTfilt_1d(sig)
        self.fs = fs
        self.rpeaks, = ecg.hamilton_segmenter(signal=self.sig, sampling_rate=self.fs)
        self.rpeaks, = ecg.correct_rpeaks(signal=self.sig, rpeaks=self.rpeaks,
                                         sampling_rate=self.fs)
        self.RR_intervals = np.diff(self.rpeaks)
        self.dRR = np.diff(self.RR_intervals)

    def __get_sdnn(self):  # 计算RR间期标准差
        return np.array([np.std(self.RR_intervals)])

    def __get_maxRR(self):  # 计算最大RR间期
        return np.array([np.max(self.RR_intervals)])

    def __get_minRR(self):  # 计算最小RR间期
        return np.array([np.min(self.RR_intervals)])

    def __get_meanRR(self):  # 计算平均RR间期
        return np.array([np.mean(self.RR_intervals)])

    def __get_medRR(self):
        return np.array([np.median(self.RR_intervals)])

    def __get__percent20th(self):
        return np.array([np.percentile(self.RR_intervals,20)])

    def __get__percent80th(self):
        return np.array([np.percentile(self.RR_intervals, 80)])

    def __get_qd(self):
        return np.array([np.percentile(self.RR_intervals,75)-np.percentile(self.RR_intervals,25)])

    def __get_Rdensity(self):  # 计算R波密度
        return np.array([(self.RR_intervals.shape[0] + 1) 
                         / self.sig.shape[0] * self.fs])

    def __get_pNN50(self):  # 计算pNN50
        return np.array([self.dRR[self.dRR >= self.fs*0.05].shape[0] 
                         / self.RR_intervals.shape[0]])

    def __get_pNN20(self):  # 计算pNN520
        return np.array([self.dRR[self.dRR >= self.fs*0.02].shape[0]
                         / self.RR_intervals.shape[0]])

    def __get_RMSSD(self):  # 计算RMSSD
        return np.array([np.sqrt(np.mean(self.dRR*self.dRR))])

    def __get_cvsd(self):
        return self.__get_RMSSD()/self.__get_meanRR()

    def __get_SampEn(self):  # 计算RR间期采样熵
        sampEn = ent.sample_entropy(self.RR_intervals, 
                                  2, 0.2 * np.std(self.RR_intervals))
        for i in range(len(sampEn)):
            if np.isnan(sampEn[i]):
                sampEn[i] = -2
            if np.isinf(sampEn[i]):
                sampEn[i] = -1
        return sampEn


    def extract_features(self):  # 提取HRV所有特征
        features = np.concatenate((self.__get_sdnn(),
                self.__get_maxRR(),
                self.__get_minRR(),
                self.__get_meanRR(),
                self.__get_medRR(),
                self.__get__percent20th(),
                self.__get__percent80th(),
                self.__get_qd(),
                self.__get_Rdensity(),
                self.__get_pNN50(),
                self.__get_pNN20(),
                self.__get_RMSSD(),
                self.__get_cvsd(),
                self.__get_SampEn(),
                ))
        assert features.shape[0] == ManFeat_HRV.FEAT_DIMENSION
        return features

