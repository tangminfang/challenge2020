#!/usr/bin/env python

import numpy as np
from scipy.signal import butter, lfilter
from scipy import stats

def detect_peaks(ecg_measurements,signal_frequency,gain):

        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.

        This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. 
        It was created and used for experimental purposes in psychophysiology and psychology.
        You can find more information in module documentation:
        https://github.com/c-labpl/qrs_detector
        If you use these modules in a research project, please consider citing it:
        https://zenodo.org/record/583770
        If you use these modules in any other project, please refer to MIT open-source license.

        If you have any question on the implementation, please refer to:

        Michal Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
        Marta lukowska (Jagiellonian University)
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/c-labpl/qrs_detector
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        
        MIT License
        Copyright (c) 2017 Michal Sznajder, Marta Lukowska
    
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

        """


        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1
        integration_window = 30  # Change proportionally when adjusting frequency (in samples).
        findpeaks_limit = 0.35
        findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).
        refractory_period = 240  # Change proportionally when adjusting frequency (in samples).
        qrs_peak_filtering_factor = 0.125
        noise_peak_filtering_factor = 0.125
        qrs_noise_diff_weight = 0.25


        # Detection results.
        qrs_peaks_indices = np.array([], dtype=int)
        noise_peaks_indices = np.array([], dtype=int)


        # Measurements filtering - 0-15 Hz band pass filter.
        filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)

        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window)/integration_window)

        # Fiducial mark - peak detection on integrated measurements.
        detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=findpeaks_limit,
                                                     spacing=findpeaks_spacing)

        detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

        return detected_peaks_values,detected_peaks_indices

 
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

def findpeaks(data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind


##########################################################################
import math
def PQST_detect(ecg, r_peaks, fs):
    V = 0.18;
    p_index = np.array([], dtype=int);
    q_index = np.array([], dtype=int);
    S_index = np.array([], dtype=int);
    t_index = np.array([], dtype=int);

    # 定义两个时间阶段
    delta_t1 = int(0.05 * fs);
    delta_t2 = int(0.2 * fs);
    ##提取Q点部分
    # 检查r前0.08s有没有超出索引
    r_peaks_Q = r_peaks;
    if r_peaks_Q[0] - fs * 0.2 < 0:
        r_peaks_Q = np.delete(r_peaks_Q, 0);
    [points, ] = r_peaks_Q.shape;
    for n in range(points):
        q_min1 = np.min(ecg[(r_peaks_Q[n] - delta_t1):r_peaks_Q[n]]);
        q_min2 = np.min(ecg[(r_peaks_Q[n] - delta_t2):r_peaks_Q[n]]);
        q_index1 = np.argmin(ecg[(r_peaks_Q[n] - delta_t1):r_peaks_Q[n]]);
        q_index2 = np.argmin(ecg[(r_peaks_Q[n] - delta_t2):r_peaks_Q[n]]);
        # 修正Q_index的值
        q_index1 = q_index1 + (r_peaks_Q[n] - delta_t1);
        q_index2 = q_index2 + (r_peaks_Q[n] - delta_t2);

        # 确定Q点
        if q_index1 == q_index2:
            q_index = np.append(q_index, q_index1);
        else:
            Max_Q = np.max(ecg[q_index2:q_index1]);
            if Max_Q > q_min1 + V:
                q_index = np.append(q_index, q_index2);
            else:
                if q_min1 >= q_min2:
                    q_index = np.append(q_index, q_index2);
                else:
                    q_index = np.append(q_index, q_index1);

    # 提取S点部分
    [ecg_points, ] = ecg.shape;
    r_peaks_S = r_peaks;
    [points, ] = r_peaks_S.shape;
    if r_peaks_S[points - 1] + fs * 0.2 > (ecg_points - 1):
        r_peaks_S = np.delete(r_peaks_S, points - 1);
    [points, ] = r_peaks_S.shape;
    for n in range(points):
        S_min1 = np.min(ecg[r_peaks_S[n]:(r_peaks_S[n] + delta_t1)]);
        S_min2 = np.min(ecg[r_peaks_S[n]:(r_peaks_S[n] + delta_t2)]);
        S_index1 = np.argmin(ecg[r_peaks_S[n]:(r_peaks_S[n] + delta_t1)]);
        S_index2 = np.argmin(ecg[r_peaks_S[n]:(r_peaks_S[n] + delta_t2)]);
        S_index1 = S_index1 + r_peaks_S[n];
        S_index2 = S_index2 + r_peaks_S[n];
        if S_min1 >= S_min2:
            S_index = np.append(S_index, S_index1);
        else:
            S_index = np.append(S_index, S_index2);

    [points, ] = r_peaks.shape;

    ###该行以上已经完成调试
    # 提取P波和T波
    for n in range(points - 1):
        RR = r_peaks[n + 1] - r_peaks[n]
        # T波峰值位置为R-R间隔17%-50%时间段内的最大值，P波峰值位置为R-R间隔75%-83.3%时间段内的最大值
        T_field = range((r_peaks[n] + math.floor(RR * 0.17)), (r_peaks[n] + math.ceil(RR * 0.5)), 1)
        P_field = range((r_peaks[n] + math.floor(RR * 0.75)), (r_peaks[n] + math.ceil(RR * 0.833)), 1)
        T_index = np.argmax(ecg[T_field]) + r_peaks[n] + math.floor(RR * 0.17)
        P_index = np.argmax(ecg[P_field]) + r_peaks[n] + math.floor(RR * 0.75)
        p_index = np.append(p_index, P_index)
        t_index = np.append(t_index, T_index)

    return [p_index, q_index, S_index, t_index]


#############################################################################
from pyentrp import entropy as ent


def get_SampEn(idx):  # 计算RR间期采样熵
    sampEn = ent.sample_entropy(idx, 2, 0.2 * np.std(idx))
    for i in range(len(sampEn)):
        if np.isnan(sampEn[i]):
            sampEn[i] = -2
        if np.isinf(sampEn[i]):
            sampEn[i] = -1
    return sampEn

def get_12ECG_features(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            label = iline.split(': ')[1].split(',')[0]


    
#   We are only using data from lead1
    peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[0])
   
#   mean
    mean_RR = np.mean(idx/sample_Fs*1000)
    mean_Peaks = np.mean(peaks*gain_lead[0])

#   median
    median_RR = np.median(idx/sample_Fs*1000)
    median_Peaks = np.median(peaks*gain_lead[0])

#   standard deviation
    std_RR = np.std(idx/sample_Fs*1000)
    std_Peaks = np.std(peaks*gain_lead[0])

#   variance
    var_RR = stats.tvar(idx/sample_Fs*1000)
    var_Peaks = stats.tvar(peaks*gain_lead[0])

#   Skewness
    skew_RR = stats.skew(idx/sample_Fs*1000)
    skew_Peaks = stats.skew(peaks*gain_lead[0])

#   Kurtosis
    kurt_RR = stats.kurtosis(idx/sample_Fs*1000)
    kurt_Peaks = stats.kurtosis(peaks*gain_lead[0])

    ###################################################################
    dRR = np.diff(idx)

    RRmax = np.max(idx/sample_Fs*1000)

    RRmin = np.min(idx/sample_Fs*1000)

    percent20th = np.percentile(idx/sample_Fs*1000, 20)

    percent80th = np.percentile(idx/sample_Fs*1000, 80)

    qd = np.percentile(idx/sample_Fs*1000, 75) - np.percentile(idx/sample_Fs*1000, 25)
    # # 计算R波密度
    # Rdensity =(idx.shape[0] + 1)/ data[0].shape[0] * sample_Fs

    pNN50 = dRR[dRR >= sample_Fs * 0.05].shape[0]/idx.shape[0]

    pNN20 = dRR[dRR >= sample_Fs * 0.02].shape[0]/idx.shape[0]

    RMSSD = np.sqrt(np.mean(dRR * dRR))

    cvsd = RMSSD / mean_RR

    sampEn= get_SampEn(idx)

    ####################################################################################
    # [p_peaks, q_peaks, s_peaks, t_peaks] = PQST_detect(data[0], idx, sample_Fs);
    # # 事先定义好会提取的特征点数
    #
    #
    # # PQ、QS、ST、TP的均值，标准差，中位数
    # [num, ] = data[0].shape;
    # [p_num, ] = p_peaks.shape;
    # [q_num, ] = q_peaks.shape;
    # [r_num, ] = idx.shape;
    # [s_num, ] = s_peaks.shape;
    # [t_num, ] = t_peaks.shape;
    # # 求PQ的间距
    # if abs(p_num - q_num) == 1:
    #     q_temp = np.delete(q_peaks, 0);
    # else:
    #     q_temp = q_peaks;
    # pq_interval = q_temp - p_peaks;
    #
    # # 求QS间距
    # if idx[0] - math.ceil(0.2 * sample_Fs) < 0:
    #     s_temp = np.delete(s_peaks, 0);
    # else:
    #     s_temp = s_peaks;
    #
    # if idx[r_num - 1] + math.ceil(0.2 * sample_Fs) >= num:
    #     q_temp = np.delete(q_peaks, q_num - 1);
    # else:
    #     q_temp = q_peaks;
    # qs_interval = s_temp - q_temp;
    #
    # # 求ST间距
    # if s_num - t_num == 1:
    #     s_temp = np.delete(s_peaks, 0);
    # else:
    #     s_temp = s_peaks;
    # st_interval = t_peaks - s_temp;
    # # 求TP间距
    # tp_interval = p_peaks - t_peaks;
    #
    # interval = [pq_interval, qs_interval, st_interval, tp_interval]
    # # pqst=[]
    # # for i in range(4):
    # #     pqst = np.append(pqst,np.mean(interval[i]))
    # #     pqst = np.append(pqst,np.std(interval[i]))
    # #     pqst = np.append(pqst,np.median(interval[i]))
    #########################################################################################

    features = np.hstack([age,sex,mean_RR,mean_Peaks,median_RR,median_Peaks,std_RR,std_Peaks,var_RR,var_Peaks,skew_RR,skew_Peaks,
                          kurt_RR,kurt_Peaks,RRmax,RRmin,percent20th,percent80th,qd,pNN20,pNN50,RMSSD,cvsd,sampEn])
    # for i in range(4):
    #     features = np.append(features,np.mean(interval[i]))
    #     features = np.append(features,np.std(interval[i]))
    #     features = np.append(features,np.median(interval[i]))

    return features


