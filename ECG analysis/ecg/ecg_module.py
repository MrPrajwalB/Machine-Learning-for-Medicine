import numpy as np
import pandas as pd
from .ecg_utils import *
import scipy.signal as ss
from sklearn.externals import joblib as jl
from biosppy.signals import ecg as ecgsig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Signal():
    def __init__(self, data, fs, hpass_cor_freq_hz=0.5, lpass_cor_freq_hz=40):
        self.data = np.array(data)
        self.fs = fs
        self._hpass_cor_freq_hz = hpass_cor_freq_hz
        self._lpass_cor_freq_hz = lpass_cor_freq_hz
        
        # Preprocess the data
        self.df = make_ecg_dataframe(self.data, self.fs, col_name = 'ecg')
        self.df, self.rpeaks = process_ecg_df(self.df, fs, lp_cornerfreq = self._lpass_cor_freq_hz, 
                                              input_col = 'ecg', output_col = 'processed', 
                                              get_rpeaks = True, rpeak_col = 'r_peak')
        self.processed = self.df["processed"].values
        
    def get_heartrate(self, unit='permin'):
        self.heartrate = heart_rate(self.rpeaks, self.fs, unit=unit)
        return self
    
    def get_heartratevar(self, unit='ms'):
        self.heartratevar = heart_rate_var(self.rpeaks, self.fs, unit=unit)
        return self    
  
    # Divide whole signal into segments
    def add_segment_id(self, window, trunc_end = True):
        index_vals = self.df.index.values
        segments = divide_segments(index_vals, window)
        segment_id = np.hstack((np.ndarray.flatten(np.repeat(np.arange(segments.shape[0]).reshape(-1, 1), 
                                                             window, axis = 1)), 
                                np.ones(len(index_vals) - np.prod(segments.shape))*(segments.shape[0])))
        self.df['segment_id'] = np.array(segment_id, dtype = int)
        
        # Compile all segments in a single dataframe
        segments_df = pd.DataFrame(self.df.groupby('segment_id')['processed'].apply(list))
        
        if len(segments_df.iloc[-1]['processed']) < window:
            if trunc_end:
                print('Last segment is shorter than window size was removed.')
                self.segments_df = pd.DataFrame(segments_df.iloc[:-1])
            else:
                print('Last segment is shorter than window size was retained.')
                self.segments_df = pd.DataFrame(segments_df)
        else:
            self.segments_df = pd.DataFrame(segments_df)        
        return self
    
    def get_segment_feats(self, beat_duration = 100e-3, rate_unit = 'permin',
                            get_beats = True):
        df_ = self.segments_df.copy()
        self.seg_feats=self.segments_df.copy()
        # R peaks
        seg_rpeaks = df_.processed.apply(lambda x: r_peak_loc(np.array(x), self.fs))
        rr_interval=df_.processed.apply(lambda x: rr_int(np.array(x), self.fs))
        self.seg_feats['r_peaks'] = seg_rpeaks
        self.seg_feats['rr_int'] = rr_interval

        # Heart rate and heart rate variability
        seg_hr = [heart_rate(x, self.fs, unit=rate_unit) for x in seg_rpeaks]
        seg_hrv = [heart_rate_var(x, self.fs) for x in seg_rpeaks]
        hrv_ind= [calculate_hrv_features(x, self.fs) for x in rr_interval]
        self.seg_feats['hr_permin'] = seg_hr        
        self.seg_feats['hrv_ms'] = seg_hrv
        self.seg_feats['HRV_ind_15']= hrv_ind
        # Beats
        if get_beats:
            seg_beats = [ecg_beats(np.array(df_.ix[i].processed), seg_rpeaks[i], 
                               self.fs, dt=beat_duration) for i in df_.index.values]
            self.seg_feats['rpeak_beat_pair'] = seg_beats
            self.seg_feats['beats'] = [ecg_beatsstack(np.array(df_.ix[i].processed), seg_rpeaks[i], 
                               self.fs, dt=beat_duration) for i in df_.index.values]                                        
            beats_rp = []
            beats_mag = []
            for b in seg_beats:
                if len(b)>0:
                    beatrp, beatmag = map(list, zip(*b))
                else:
                    beatrp = []
                    beatmag = []
                beats_rp.append(beatrp)
                beats_mag.append(beatmag)
            self.beats_df = pd.DataFrame({'segment_id': self.seg_feats.index.values,
                                         'beat_rpeak': beats_rp, 'beat_mag': beats_mag}).set_index('segment_id')
        final_feats=get_features(self.seg_feats, self.fs)
        return final_feats
