# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 13:51:10 2018

@author: Winham

# CPSC_config.py: 相关参数配置

"""
import math

class Config(object):
    def __init__(self):
        # 随机数种子
        self.RANDOM_STATE = 42
        
        # 导联数目
        self.LEAD_NUM = 12
        
        # 各类名称
        self.CLASS_NAME = ['N', 'AF', 'AVB', 'LBBB',
        'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        
        # 数据存放路径
        self.DATA_PATH = '/home/zyhk/桌面/DataSet_250Hz/'
        
        # 标签存放路径
        self.REVISED_LABEL = '/home/zyhk/桌面/CPSC_Scheme-master/Record_Label.npy'
        
        # keras深度模型存放路径
        self.MODEL_PATH = '/home/zyhk/桌面/CPSC_Scheme-master/model_torg/'
        
        # 人工特征存放路径
        self.MAN_FEATURE_PATH = '/home/zyhk/桌面/CPSC_Scheme-master/Man_features/'
        
        # 个体年龄性别信息存放路径
        self.AGE_GEN_INFO = '/home/zyhk/桌面/CPSC_Scheme-master/info_age_gen.csv'
        
        # 信号采样率
        self.Fs = 250
        
        # 信号切片数目
        self.SEG_NUM = 24
        
        # 信号切片时间长度
        self.SEG_TIME_LENGTH = 6.0
        
        # 信号采样点长度
        self.SEG_LENGTH = int(self.Fs * self.SEG_TIME_LENGTH)
    
    @staticmethod
    def lr_schedule(epoch):
        # 训练网络时学习率衰减方案
        
        # lr = 0.1
        # if epoch >= 20 and epoch < 60:
        #     lr = 0.01
        # if epoch >= 60:
        #     lr = 0.001
        # print('Learning rate: ', lr)
        # return lr
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop,math.floor((1 + epoch) / epochs_drop))
        return lrate
