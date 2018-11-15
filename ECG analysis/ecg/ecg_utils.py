from __future__ import division
import numpy as np
import pandas as pd
import scipy.signal as ss
import itertools
from biosppy.signals import ecg as ecgsig
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
import wfdb.processing as wp
from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear

## process ECG signal 

def baseline_correct(x, fs):
    """
    Removes baseline wander of ECG signal
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
    
    Returns
    -------
    corrected: array
        Array of containing the baseline-corrected signal
    
    """
    
    med_filt = ss.medfilt(ss.medfilt(x, kernel_size=int(200e-3*fs+1)), kernel_size = int(600e-3*fs+1))
    corrected = x-med_filt
    return corrected

    
def smooth(x, fs, order = 1, btype = 'low', corner_freq_hz = 150):
    """
    Smoothens signal using Butterworth filter
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    order: int, optional
        The order of the filter

    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, optional
        The type of filter.  Default is 'lowpass'.
    
    corner_freq_hz: scalar
        Number corresponding to the critical frequency of the filter
        
    Returns
    -------
    filtered: array
        Smoothened signal
        
    """
    nyquist = fs / 2.0
    f_c = np.array([corner_freq_hz, ], dtype=np.float64)  # Hz
    # Normalize by Nyquist
    f_c /= nyquist
    # Second order Butterworth low pass filter at corner frequency
    b, a = ss.butter(order, f_c, btype=btype)
    # Apply the filter forward and backward to eliminate delay.
    filtered = ss.filtfilt(b, a, x)
    return filtered


def preprocess_ecg(x, fs, lp_cornerfreq = 40):
    """
    Performs pre-processing of the raw ECG signal (High pass filter, 
    Low pass filter and Baseline correction)
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
                
    lp_cornerfreq: scalar, optional
        Number corresponding to the corner frequency of the filter
        
    Returns
    -------
    x3: array
        Processed signal
    """

    x1 = smooth(x, fs, btype = 'high', corner_freq_hz = 0.5)
    x2 = smooth(x1, fs, corner_freq_hz = lp_cornerfreq)
    x3 = baseline_correct(x2, fs)
    return x3


def r_peak_loc(x, fs):
    """
    Gives location of R peaks
    
    Parameters
    ----------
    x : array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
    
    Returns
    -------
    rloc: array
        Array containing the R-peak locations in the array
        
    """
    rloc = ecgsig.hamilton_segmenter(x, fs)['rpeaks']
    return rloc

def rr_int(x,fs):
    rpeaks=np.asarray(list(r_peak_loc(x,fs)))
    rr_inte=wp.calc_rr(rpeaks, fs=fs)
    return rr_inte


def ecg_beatsstack(sig, rpeaks, fs, dt = 100e-3):
    """
    Gives an array of ECG beats
    
    Parameters
    ----------
    sig: array_like
        Array containing magnitudes of ECG signal

    rpeaks: array_like
        Array containing the index of the R peaks in the signal
    
    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    dt: float, optional
        Number corresponding to the time duration to be taken from to left 
        and to the right of the R peak location, must be in seconds
    
    Returns
    -------
    beats: array
        Contains the ECG beats where the magnitudes of each beat is given by
        every row
        
    """
    width = int(dt * fs)
    beats = np.array([sig[int(r - width): int(r + width)] for r in rpeaks if ((r - width) >0 and (r + width)< len(sig))])
    return beats


def make_ecg_dataframe(x, fs, col_name = 'ecg'):
    """
    Converts an array of ECG signal to a data frame
    
    Parameters
    ----------
    x: array_like
        Array containing magnitudes of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
    
    col_name: string, optional
        String that will be used as a column name in the data frame
        
    Returns
    -------
    df: pandas dataframe
        Daframe containing the magnitudes of the ECG signal in one columns
        and the corresponding time of in seconds
   
    """
    df = pd.DataFrame({col_name: x, '_time_sec':np.arange(len(x))/fs})
    return df


def process_ecg_df(df, fs, lp_cornerfreq = 40, input_col = 'ecg', output_col = 'processed', 
                   get_rpeaks = False, rpeak_col = 'r_peak_loc'):
    """
    Adds the processed ECG signal and the R peaks (optional) to the dataframe 
    
    Parameters
    ----------
    df: pandas data frame
        data frame containing the magnitudes of the raw ECG signal in one column

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    lp_cornerfreq: scalar, optional
        Number corresponding to the corner frequency of the filter to be applied   
        
    input_col: string, optional
        String that corresponds to the column name of raw ECG signal
        
    output_col: string, optional
        String that will be used as a column name in the data frame for the 
        processed signal
        
    get_rpeaks: bool, optional
        If this is set to True, the R peaks in the signal will be labeled as 1
        and others as 0
        
    rpeak_col: string, optional
        String that will be used as a column name in the data frame for the 
        R peaks
    
    """
    df[output_col] = preprocess_ecg(df[input_col], fs, lp_cornerfreq)
    sig = df[output_col].values
    if get_rpeaks:
        df[rpeak_col] = 0
        rpeak_index = r_peak_loc(sig, fs)
        df.loc[rpeak_index, rpeak_col] = 1
        return df, rpeak_index
    else:
        return df
                        
    
def heart_rate(r_peaks, fs, unit = 'persec'):
    """
    Calculate heart rate from the location of R peaks
    
    Parameters
    ----------
    rpeaks: array_like
        Array containing indices of the R peaks in the ECG signal  
    
    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    unit: string, optional 
        Specifies unit of heart rate ('persec' refers to breaths per second,
        and 'permin' refers to breaths per minute)
        
    Returns
    -------
    hr: float
        Estimate of the heart rate
        
    """ 
    multiplier = {'persec': 1, 'permin': 60}
    time_peaks = r_peaks/fs
    dt = time_peaks[1:] - time_peaks[:-1]
    hr = 1/np.nanmean(dt) *multiplier[unit]
    return hr

def heart_rate_var(r_peaks, fs, unit = 'ms'):
    """
    Calculate heart rate from the location of R peaks
    
    Parameters
    ----------
    rpeaks: array_like
        Array containing indices of the R peaks in the ECG signal  
    
    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
        
    unit: string, optional 
        Specifies unit of heart rate ('s' refers to second,
        and 'ms' refers to millisecond)
        
    Returns
    -------
    hrv: float
        Estimate of the heart rate variability
        
    """     
    multiplier = {'s': 1, 'ms': 1000}    
    time_peaks = r_peaks/fs
    dt = time_peaks[1:] - time_peaks[:-1]
    hrv = np.nanstd(dt)*multiplier[unit]
    return hrv
    
## dividing the whole signal into segments
def array_rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def divide_segments(a, window, overlap = 0):
    window = int(window)
    rolled = array_rolling_window(a, window)
    slide_by = int(window*(1-overlap))
    return rolled[0::slide_by]

def ecg_beats(sig, rpeaks, fs, dt = 100e-3):
    width = int(dt * fs)
    beats = [(r, sig[int(r - width): int(r + width)]) for r in rpeaks 
             if ((r - width) >0 and (r + width)< len(sig))]
    return beats



##############################################################
##############################################################
##############################################################

def iqr(x):
    """
    Calculate inter-quartile range
    
    Parameters
    ----------
    x: array_like
        Array containing numbers for IQR
        
    Returns
    -------
    iqr: float
        Inter-quartile Range
        
    """ 
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    return iqr

def mad(x):
    """
    Calculate mean absolute deviation
    
    Parameters
    ----------
    x: array_like
        Array containing numbers for MAD
        
    Returns
    -------
    mad: float
        mean absolute deviation
        
    """ 
    
    mean_x = np.mean(x)
    mean_adj = np.abs(x - mean_x)
    mad = np.mean(mean_adj)
    return mad

def sample_entropy(a):
    """
    Calculate 2nd order sample entropy
    
    Parameters
    ----------
    a: array_like
        Array containing numbers to calculate sample entropy
        
    Returns
    -------
    sp: float
        sample entropy
        
    """ 
    sp = np.abs(a[2] - a).max(axis=0)
    return sp

def segment_ecg(ecg, segment_size = 60, f = 100):
    """
    Segment ECG signal
    
    Parameters
    ----------
    ecg: array_like
        1D Array containing amplitude of ECG signal
        
        
    segment_size: int, optional 
        Specifies segment length (15-sec, 30-sec, or 60-sec)
        
    f: float, optional
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
       
    Returns
    -------
    ecg_segmented: array_like
        Estimate of the heart rate variability
        
    """ 
    
    ecg[ecg < 0.01] = 0.01
    ecg_len = ecg.shape
    divs =int(ecg_len[0] /(segment_size * f))
    ecg_new = baseline_correct(ecg, f)
    windowed_ecg = np.array_split(ecg_new[:divs * segment_size * f], divs)  
    ecg_segmented = np.array([i for i in windowed_ecg if len(i) == len(windowed_ecg[0])])
    return ecg_segmented
    
def calculate_hrv_features(rri, f = 360):
    """
    Calculate features for detecting sleep apnea
    
    The features contain statistical measures, morphology and periodogram
    of the ECG signal.
       
    Parameters
    ----------
    RR_int: array_like
        Array containing RR intervals
        
    f: float, optional
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
           
    Returns
    -------
    X: array_like
        Features
        
    """ 
    result_td= np.reshape(np.asarray(list(time_domain(rri).values())),(1,6))
    result_fd= np.reshape(np.asarray(list(frequency_domain(rri).values())),(1,7))
    result_nl=np.reshape(np.asarray(list(non_linear(rri).values())), (1,2))
    hrv_features=np.concatenate([result_td, result_fd, result_nl], axis =1)
    return hrv_features


## signal quality features


def rel_power(ecg_sig, fs, num_freqbounds = (5, 15), denom_freqbounds = (5, 40)):
    """
    Computes relative power of a signal
    
    Parameters
    ----------
    ecg_sig: array_like
        Array containing magnitude of ECG signal    

    fs: float
        Sampling rate of signal (must be in Hertz)
    
    num_freqbounds: tuple, optional
        tuple of frequencies (in Hertz) corresponding to the bandwidth of interest
        
    denom_freqbounds: tuple, optional
        tuple of frequencies (in Hertz) corresponding to the bandwidth of interest        
    
    Returns
    -------
    relpwr: float 
        relative power contained in the range of frequencies with respect to the 
        range of denominator frequencies
    """
            
    powerspec = ss.periodogram(ecg_sig, fs)
    numerator = np.trapz((powerspec[1])[(powerspec[0] >= num_freqbounds[0]) * (powerspec[0] <=num_freqbounds[1])], 
                         dx = powerspec[0][1] - powerspec[0][0])
    denominator = np.trapz((powerspec[1])[(powerspec[0] >= denom_freqbounds[0]) * (powerspec[0] <=denom_freqbounds[1])], 
                           dx = powerspec[0][1] - powerspec[0][0])    
    relpwr = numerator/denominator
    return relpwr

def power_spec(ecg_sig, fs, bins = 5, fmax = 5):
    """
    
    Parameters
    ----------
    ecg_sig: array_like
        Array containing magnitude of ECG signal    

    fs: float
        Sampling rate of signal (must be in Hertz)

    bins: int, optional
        Number of frequency bins
        
    fmax: float, optional
        Maximum frequency in the power spectrum
        
    Returns
    -------
    pwrspec: array_like
        array of magnitudes of power spectrum for the frequency bins set
    """
        
    ecg_sig -= np.nanmean(ecg_sig)
    a, b = ss.periodogram(ecg_sig, fs = fs)
    b[0] = 0
    b = b/np.max(b)
    b = b[a<fmax] 
    a = a[a<fmax]
    
    pwrspec = np.sum(np.reshape(b, (bins, -1)), axis = 1) 
    return pwrspec

def sig_energy(ecg_sig):
    """
    Computes the energy of the signal
    
    Parameters
    ----------
    ecg_sig: array_like
        Array containing magnitude of ECG signal
    
    Returns
    -------
    energy: float
        Number that refers to the signal energy
    """
            
    ecg_sig -= np.nanmean(ecg_sig)
    ecg_sig /= np.max(abs(ecg_sig))
    energy = np.sum(np.square(ecg_sig))
    return energy

def permutation_entropy(time_series, m, delay):
    """
    Calculates permutation entropy of a time series
    
    Parameters
    ----------
    time_series: array_like
        Time series signal
        
    m: int
        Determines number of accessible states
        
    delay: float
        time lag
    
    Returns
    -------
    pe: float
        value of permutation entropy of the time series
    """
            
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    c = [0] * len(permutations)

    for i in range(n - delay * (m - 1)):
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
        for j in range(len(permutations)):
            if abs(permutations[j] - sorted_index_array).any() == 0:
                c[j] += 1

    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe

def normal_hr(sig, fs=360):
    """
    Returns the estimate of normal heart rate
   
    Parameters
    ----------
    sig: array_like
        Contains magnitude of ECG signal

    fs: float
        Number corresponding to the sampling frequency of the input signal,
        must be in Hertz
                 
    Returns
    -------
    fnhr: float 
        Normal heart rate value
    """
    stdv = np.std(sig)
    samp_clip = np.clip(sig, -stdv*0.7, stdv*0.7)
    samp_smt = smooth(samp_clip, fs, order = 6, corner_freq_hz=2)
    
    #first derivative
    ss_fd = ss.savgol_filter(samp_smt, window_length=7, polyorder=5, deriv=1)

    #power spectral density
    psd_1, psd_2 = ss.periodogram(ss_fd, fs)
    fnhr = psd_1[np.argmax(psd_2)]
    return fnhr

def rtor_duration(r_peaks, fs, unit = 'ms'):
    """
    Returns the mean duration between two R peaks
    
    Parameters
    ----------
    r_peaks: array_like
        Contains magnitude of ECG beats where each row is a beat

    fs: float
        Sampling rate of signal (must be in Hertz)
       
    unit: string, optional 
        Specifies unit of heart rate ('s' refers to second,
        and 'ms' refers to millisecond)
        
    Returns
    -------
    rtor: float
        Mean duration between two R peaks
    
    """          
    multiplier = {'s': 1, 'ms': 1000}    
    time_peaks = r_peaks/fs
    dt = time_peaks[1:] - time_peaks[:-1]
    rtor = np.mean(dt)*multiplier[unit]
    return rtor


down_sampler = pd.read_pickle("data//signal_quality//pca_200msbeat.p")

def pca_feature(beats, top_comp = 5):
    """
    Returns the mean of the top principal components of ECG beats
    
    Parameters
    ----------
    beats: array_like
        Contains magnitude of ECG beats where each row is a beat
    
    top_comp: int, optional
        Number of principal components
        
    Returns
    -------
    pca_comps: array_like
        Array containing the mean of the top principal components
    
    """
    
    beat_length = beats.shape[1]
    if beat_length < down_sampler.n_components:
        f_out = interp1d(np.arange(beat_length), beats, axis=1)
        beats = f_out(np.linspace(0, beat_length-1, down_sampler.n_components))
    
    beats_pca = down_sampler.transform(beats)
    pca_comps = np.mean(np.sum(beats_pca[:, :top_comp], axis = 1)/np.sum(beats_pca, axis = 1))
    return pca_comps

def mean_beat_energy(beats):
    """
    Computes the mean energy of beats
    
    Parameters
    ----------
    beats: array_like
        Contains magnitude of ECG beats where each row is a beat
     
    Returns
    -------
    mean_energy: float
        Mean of all beat energy
    
    """    
    beats /= np.max(np.ndarray.flatten(beats))
    energy = np.square(beats).sum(axis = 1)
    mean_energy = np.mean(energy)
    return mean_energy

def rms(x):
    """
    Computes root-mean-square value of signal
    
    Parameters
    ----------
    x: array_like
        Contains magnitude of ECG signal
     
    Returns
    -------
    rms_val: float
        root-mean-square value of input signal
    
    """
    rms_val = np.sqrt(np.mean(np.square(x)))
    return rms_val

def maxmin_beat(beats):
    """
    Computes mean beat amplitude
    
    Parameters
    ----------
    beats: array_like
        Contains magnitude of ECG beats where each row is a beat
     
    Returns
    -------
    maxmin: float
        Mean beat amplitude
    
    """
    beats /= np.max(np.ndarray.flatten(beats))
    maxmin = np.mean(np.max(beats, axis=1) - np.min(beats, axis=1))
    return maxmin

def sum_beat_energy(beats):
    """
    Computes the sum of energy of beats
    
    Parameters
    ----------
    beats: array_like
        Contains magnitude of ECG beats where each row is a beat
     
    Returns
    -------
    sumbe: float
        Sum of all beat energy
    
    """
    beats /= np.max(np.ndarray.flatten(beats))
    sumbe = np.sum(np.square(np.sum(beats, axis=0))) 
    return sumbe


def get_features(df_, fs):
    """
    Calculates features to be used for ECG signal quality classification
    
    Parameters
    ----------
    df_: pandas dataframe
        Dataframe of ECG data, must contain the following columns: 
        processed, r_peaks, and beats

    fs: float
        Sampling rate of signal (must be in Hertz)
        
     
    Returns
    -------
    df: pandas dataframe
        Dataframe appended with computed features
    """
    
    
    df = pd.DataFrame.copy(df_)
    
    print('Computing features...'),
    # features from statistics of magnitude of ECG signal
    df.loc[:, 'f_stddev'] = df.processed.apply(lambda x: np.nanstd(x))
    df.loc[:, 'f_kurtosis'] = df.processed.apply(lambda x: kurtosis(x))
    df.loc[:, 'f_skewness'] = df.processed.apply(lambda x: skew(x))
    df.loc[:, 'f_rms'] = df.processed.apply(lambda x: rms(x))
    df.loc[:, 'f_energy'] = df.processed.apply(lambda x: sig_energy(x))

    # features from power spectrum of signal
    df.loc[:, 'f_relpower'] = df.processed.apply(lambda x: rel_power(x, fs))
    df.loc[:, 'f_relbasepower'] = df.processed.apply(lambda x: rel_power(x, fs, num_freqbounds=(1, 40), 
                                                                  denom_freqbounds = (0, 40)))
    fbins, fmax = 10, 10
    powspec_vals = np.vstack(df.processed.apply(lambda x: power_spec(x, fs, bins=fbins, fmax=fmax)).values)
    for i in range(fbins):
        df.loc[:, 'f_powspec'+str(i)] = list(powspec_vals[:, i])

    
    # features from physiological parameters
    df.loc[:, 'f_rpeakcount'] = df.r_peaks.map(len)
    df.loc[:, 'f_nhr'] =  df.processed.apply(lambda x: normal_hr(x, fs))
    df.loc[:, 'f_hrv'] =  df.r_peaks.apply(lambda x: heart_rate_var(x, fs))
    df.loc[:, 'f_rtor'] =  df.r_peaks.apply(lambda x: rtor_duration(x, fs))
    df.loc[:, 'f_sumbe'] = df.beats.apply(lambda x: sum_beat_energy(np.array(x)))
        
    df.loc[:, 'f_pca'] = 0
    df.loc[df.beats.map(len)>0, 'f_pca'] = df.beats[df.beats.map(len)>0].apply(lambda x:pca_feature(np.array(x)))
    
#    df.loc[:, 'f_mbe'] = 0
    df.loc[df.beats.map(len)>0, 'f_mbe'] = df.beats[df.beats.map(len)>0].apply(lambda x: mean_beat_energy(np.array(x)))
    
    df.loc[:, 'f_maxminbeat'] = 0
    df.loc[df.beats.map(len)>0, 'f_maxminbeat'] = df.beats[df.beats.map(len)>0].apply(lambda x: maxmin_beat(np.array(x)))
     
    print('Done!')

    return df

