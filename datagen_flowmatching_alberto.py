from email import parser
import gc
import fix_cupy
import numpy as np
import h5py
import argparse
import torch
import json
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d

from astropy.cosmology import Planck18

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import PC_SI, YRSID_SI
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity
from ssqueezepy import ssq_cwt
from superlet import superlets

try:
    import cupy as cp
    use_gpu_available = True
except ImportError:
    use_gpu_available = False

try:
    from nnAudio.features.cqt import CQT1992v2
    has_nnaudio = True
except ImportError:
    has_nnaudio = False

from scipy.stats import gaussian_kde

def generate_astrophysical_prior(catalog_params, num_samples):
    """
    Learns the continuous multi-dimensional distribution of the empirical catalog
    and generates `num_samples` new, unique events drawn from that distribution.
    """
    print(f"-> Fitting Kernel Density Estimator to {len(catalog_params)} catalog events...")
    
   
    log_m1 = np.log10(catalog_params[:, 0])
    q = catalog_params[:, 1]
    log_dist = np.log10(catalog_params[:, 2])
    inc = catalog_params[:, 4] 
    
    
    training_data = np.vstack([log_m1, q, log_dist, inc])
    
    
    kde = gaussian_kde(training_data)
    
    
    print(f"-> Sampling {num_samples} new events from the continuous prior...")
    samples = kde.resample(num_samples)
    
    samples[0, :] = np.clip(samples[0, :], np.min(log_m1), np.max(log_m1)) # Clamp Mass
    samples[1, :] = np.clip(samples[1, :], 0.01, 1.0)                      # Clamp q
    samples[2, :] = np.clip(samples[2, :], np.min(log_dist), np.max(log_dist)) # Clamp Dist
    samples[3, :] = np.clip(samples[3, :], 0.0, np.pi)
    
    new_params = np.zeros((num_samples, 11))
    
    new_params[:, 0] = 10**samples[0, :] 
    new_params[:, 1] = samples[1, :]     
    new_params[:, 2] = 10**samples[2, :] 
    new_params[:, 4] = samples[3, :]     
    
    # sky localization and orientation - UNUIFORM IN THE SKY!!
   
    new_params[:, 5] = np.random.uniform(0, 2*np.pi, num_samples)
    new_params[:, 6] = np.arcsin(np.random.uniform(-1, 1, num_samples)) 
    new_params[:, 7] = np.random.uniform(0, np.pi, num_samples)
    new_params[:, 8] = np.random.uniform(0, 2*np.pi, num_samples)
    

    # FIX THE SPINS TO ZERO!

    new_params[:, 9] = 0.0
    new_params[:, 10] = 0.0 
    return new_params


# ======================================================================
# ALBERTO'S JSON PARSER
# ======================================================================
def extract_alberto_json(filepath):
    """ Reads Alberto's lisabeta JSON files using the exact keys discovered """
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    p = data['source_params']

    mc = float(p['Mchirp'])
    q_json = float(p['q'])
    
    m_A = mc * (1.0 + q_json)**0.2 / (q_json**0.6)       #### ARE WE SURE THIS IS THE CORRECT WAY FROM MC AND Q TO M1?
    m_B = m_A * q_json
    
    m1 = max(m_A, m_B)
    m2 = min(m_A, m_B)
    q_final = m2 / m1 
    

    dist_mpc = float(p['dist'])
    dist = dist_mpc * 1e6 * PC_SI
        
    t_ref = float(p['Deltat'])   # IS THIS TREF??  
    
    inc = float(p['inc'])
    lam = float(p['lambda'])
    beta = float(p['beta'])
    psi = float(p['psi'])
    phi_ref = float(p['phi'])
    chi1 = float(p['chi1'])
    chi2 = float(p['chi2'])
    
    return[m1, q_final, dist, t_ref, inc, lam, beta, psi, phi_ref, chi1, chi2]

# def load_alberto_population(repo_path, pop_name="Pop3", filter_modes=0):
#     h5_path = os.path.join(repo_path, "data_for_Malvina.h5")
    
#     valid_stsi =[]
#     mode_map = {} 
    
#     if os.path.exists(h5_path):
#         with h5py.File(h5_path, 'r') as f:
#             if pop_name in f:
#                 pop_group = f[pop_name]
#                 for strid in pop_group.keys():
#                     event_data = pop_group[strid]
#                     stsi = int(event_data['StSi_index'][()])
                    
#                     is_1 = event_data['1mode_5p'][()] == 1
#                     is_2 = event_data['2modes_5p'][()] == 1
#                     is_8 = event_data['8modes_5p'][()] == 1
                    
#                     mode = 0
#                     if is_1: mode = 1
#                     elif is_2: mode = 2
#                     elif is_8: mode = 8
                    
#                     mode_map[stsi] = mode
                    
#                     if filter_modes in [1, 2, 8]:
#                         if mode == filter_modes:
#                             valid_stsi.append(stsi)
#                     else:
#                         valid_stsi.append(stsi)
                        
#                 print(f"-> Found {len(valid_stsi)} events matching {filter_modes}-modes in {pop_name} index.")
#             else:
#                 print(f"Warning: {pop_name} not found in {h5_path}.")
#     else:
#         print(f"Warning: {h5_path} not found.")
                
#     json_dir = os.path.join(repo_path, pop_name, "json")
#     json_files =[]
    
#     for idx in valid_stsi:
#         matches = glob.glob(os.path.join(json_dir, f"*{int(idx)}*.json"))
#         if not matches:
#             matches = glob.glob(os.path.join(json_dir, f"{int(idx)}.json"))
#         if matches:
#             json_files.append((matches[0], mode_map[int(idx)])) 
            
#     print(f"-> Extracting parameters from {len(json_files)} JSON files...")
    
#     all_params = []
#     all_modes =[]
#     for jf, expected_mode in json_files:
#         p = extract_alberto_json(jf)
#         if p is not None:
#             all_params.append(p)
#             all_modes.append(expected_mode)
            
#     return np.array(all_params), np.array(all_modes)


def load_alberto_population(repo_path, pop_name="Pop3"):
    """
    1. Loads ALL 1272 json files to train the KDE Prior.
    2. Loads the specific 148 .h5 events with their exact modes for the Test Set.
    """
    h5_path = os.path.join(repo_path, "data_for_Malvina.h5")
    json_dir = os.path.join(repo_path, pop_name, "json")
    
    all_json_files = glob.glob(os.path.join(json_dir, "*.json"))
    full_catalog_params =[]
    
    for jf in all_json_files:
        try:
            full_catalog_params.append(extract_alberto_json(jf))
        except: pass
    
    full_catalog_params = np.array(full_catalog_params)
    print(f"-> Extracted {len(full_catalog_params)} total events for KDE fitting.")
    
    test_params = []
    test_modes =[]
    
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            if pop_name in f:
                pop_group = f[pop_name]
                for strid in pop_group.keys():
                    ev = pop_group[strid]
                    stsi = int(ev['StSi_index'][()])
                    
                    # Determine Mode
                    m = 0
                    if ev['1mode_5p'][()] == 1: m = 1
                    elif ev['2modes_5p'][()] == 1: m = 2
                    elif ev['8modes_5p'][()] == 1: m = 8
                    
                    # Find matching JSON to get the angles
                    matches = glob.glob(os.path.join(json_dir, f"*{int(stsi)}*.json"))
                    if not matches:
                        matches = glob.glob(os.path.join(json_dir, f"{int(stsi)}.json"))
                        
                    if matches:
                        try:
                            p = extract_alberto_json(matches[0])
                            test_params.append(p)
                            test_modes.append(m)
                        except: pass
    
    test_params = np.array(test_params)
    test_modes = np.array(test_modes)
    print(f"-> Extracted {len(test_params)} true benchmark events with known modes.")
    
    return full_catalog_params, test_params, test_modes
# ======================================================================
# DATASET GENERATOR
# ======================================================================
class DatasetGenerator:
    def __init__(self, dt=10.0, T_obs=172800.0, T_gen=2592000.0, noise_type='psd_whitened', use_gpu=False):
        self.dt = dt
        self.fs = 1.0 / dt
        self.T_obs = T_obs
        self.df = 1/T_obs
        self.N_obs = int(T_obs / dt)
        self.T_gen = T_gen
        self.N_gen = int(T_gen / dt)
        self.noise_type = noise_type
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        self.freqs_gen = np.fft.rfftfreq(self.N_gen, dt)

        self.dist_min = Planck18.luminosity_distance(2.0).value * 1e6 * PC_SI
        self.dist_max = Planck18.luminosity_distance(4.0).value * 1e6 * PC_SI

        if self.noise_type in ['psd', 'psd_whitened']:
            freqs_short = np.fft.rfftfreq(self.N_obs, self.dt)
            f_safe = np.clip(freqs_short, 1e-5, None)
            
            psd_A = get_sensitivity(f_safe, sens_fn="A1TDISens")
            self.psd_A = cp.asarray(psd_A) if self.use_gpu else psd_A
            self.sigma_fd_A = self.xp.sqrt(self.psd_A / (4 * self.df))
            self.psd_E = self.psd_A
            self.sigma_fd_E = self.sigma_fd_A
            
    def get_mbhb_batch(self, wave_gen, batch_size, predefined_params=None):
        if predefined_params is not None:
            m1 = self.xp.array(predefined_params[:, 0])
            q = self.xp.array(predefined_params[:, 1])
            m2 = m1 * q
            dist = self.xp.array(predefined_params[:, 2])
            
            # I had to overwrite Alberto's t_ref because it was out of ,my wiindow of 30 days!!!!!!
            t_ref = self.xp.full(batch_size, self.T_gen) - self.T_obs * self.xp.random.uniform(0.1, 0.5, batch_size)
            
            inc = self.xp.array(predefined_params[:, 4])
            lam = self.xp.array(predefined_params[:, 5])
            beta = self.xp.array(predefined_params[:, 6])
            psi = self.xp.array(predefined_params[:, 7])
            phi_ref = self.xp.array(predefined_params[:, 8])
            chi1 = self.xp.array(predefined_params[:, 9])
            chi2 = self.xp.array(predefined_params[:, 10])
            
            if m1[0] == 0:
                print(f"[ERROR] m1 is 0.0. Could not find the mass!")
        else:
            m1 = 10**self.xp.random.uniform(5, 6, batch_size)
            q = self.xp.random.uniform(0.1, 1, batch_size)
            m2 = m1 * q
            dist = self.xp.random.uniform(self.dist_min, self.dist_max, batch_size)
            t_ref = self.xp.full(batch_size, self.T_gen) - self.T_obs * self.xp.random.uniform(0.1, 0.5, batch_size)
            inc = self.xp.arccos(self.xp.random.uniform(-1, 1, batch_size))
            lam = self.xp.random.uniform(0, 2*np.pi, batch_size)
            beta = self.xp.arcsin(self.xp.random.uniform(-1, 1, batch_size))
            psi = self.xp.random.uniform(0, np.pi, batch_size)
            phi_ref = self.xp.random.uniform(0, 2*np.pi, batch_size)
            chi1 = self.xp.zeros(batch_size)
            chi2 = self.xp.zeros(batch_size)

        f_ref = self.xp.zeros(batch_size) 
        modes =[(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        
        freqs_in = cp.asarray(self.freqs_gen) if self.use_gpu else self.freqs_gen
        t_obs_end_yrs = float(self.T_gen) / YRSID_SI

        wave_out = wave_gen(
            m1, m2, chi1, chi2, dist,
            phi_ref, f_ref, inc, lam, beta, psi, t_ref,
            freqs=freqs_in, modes=modes, direct=False, fill=True, 
            squeeze=False, length=self.N_gen,
            t_obs_start=0.0, t_obs_end=t_obs_end_yrs
        )
        
        wave_fd_A = wave_out[:, 0, :] 
        wave_fd_E = wave_out[:, 1, :]
        
        if self.use_gpu:
            wave_td_A = cp.fft.irfft(wave_fd_A, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            wave_td_E = cp.fft.irfft(wave_fd_E, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            params = cp.asnumpy(cp.stack([m1, q, dist, t_ref, inc, lam, beta, psi, phi_ref, chi1, chi2], axis=1))
        else:
            wave_td_A = np.fft.irfft(wave_fd_A, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            wave_td_E = np.fft.irfft(wave_fd_E, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            params = np.stack([m1, q, dist, t_ref, inc, lam, beta, psi, phi_ref, chi1, chi2], axis=1)
            
        return wave_td_A, wave_td_E, params

    def get_noise_batch(self, batch_size):
        if self.noise_type == 'gaussian': 
            noise_A = self.xp.random.normal(0, 1e-21, (batch_size, self.N_obs))
            noise_E = self.xp.random.normal(0, 1e-21, (batch_size, self.N_obs))
            return noise_A, noise_E

        def gen_psd_noise(sigma_fd, psd_arr):
            n_real = self.xp.random.normal(0, sigma_fd, (batch_size, len(sigma_fd)))
            n_imag = self.xp.random.normal(0, sigma_fd, (batch_size, len(sigma_fd)))
            n_imag[:, 0] = 0.0
            n_real[:, 0] = self.xp.random.normal(0, self.xp.sqrt(self.N_obs / (2 * self.dt) * psd_arr[0]), batch_size)
            if self.N_obs % 2 == 0:
                n_imag[:, -1] = 0.0
                n_real[:, -1] = self.xp.random.normal(0, self.xp.sqrt(self.N_obs / (2 * self.dt) * psd_arr[-1]), batch_size)
            return self.xp.fft.irfft(n_real + 1j * n_imag, n=self.N_obs, axis=-1)

        noise_A = gen_psd_noise(self.sigma_fd_A, self.psd_A)
        noise_E = gen_psd_noise(self.sigma_fd_E, self.psd_E)
        return noise_A, noise_E

    def generate(self, n_samples, wave_gen, output_path, batch_size, 
                 repr_type='stft', channels='XYZ', nperseg=1024, noverlap=768, 
                 J=8, Q=8, T=1, c1=8, ord_min=1, ord_max=5, nv=16, phase_info=False, 
                 ssq_downsample=64, output_format='real_imag', apply_log=False,
                 predefined_population=None, predefined_modes=None):
                     
        torch_device = "cuda" if self.use_gpu else "cpu"
        n_ch = 3 if channels == 'XYZ' else 2
        time_ds = int(ssq_downsample)

        if channels == 'AE' and repr_type in ['stft', 'cwt', 'cqt']:
            if output_format == 'mag_phase': 
                n_ch_out = 3    # [Mag A, Mag E, Phase Diff]
            elif output_format == 'real_imag': 
                n_ch_out = 6    # [Re(A), Im(A), Re(E), Im(E), cos(phase_diff), sin(phase_diff)]
            elif output_format == 'mag_phase_all':
                n_ch_out = 5    # [Mag A, Mag E, Phase A, Phase E, Phase Diff]
            elif output_format == 'ampl_rel_phase':
                n_ch_out = 4
            else: 
                n_ch_out = 2    # [Mag A, Mag E]
        
        elif repr_type in ['stft', 'cwt', 'cqt']:
            n_ch_out = n_ch * 2 if output_format in ['mag_phase', 'mag_phase_all', 'real_imag'] else n_ch
        else:
            n_ch_out = n_ch * 2 if phase_info else n_ch

        if repr_type == 'stft':
            hop_length = nperseg - noverlap
            window = torch.hann_window(nperseg, device=torch_device)
            dummy = torch.zeros(self.N_obs, device=torch_device)
            stft_dummy = torch.stft(dummy, n_fft=nperseg, hop_length=hop_length, window=window, center=True, return_complex=True)
            # n_freq = 256  
            freqs = np.fft.rfftfreq(nperseg, d=self.dt)

            f_min = 5e-5       # ATTENTION !! IF YOU CHANGE HERE CHANGE ALSO AT LINE 550
            f_max = 1e-2

            mask = (freqs >= f_min) & (freqs <= f_max)

            n_freq = int(mask.sum())
            #n_freq = stft_dummy.shape[0]
            n_time = stft_dummy.shape[-1]

            
        elif repr_type == 'cqt':
            # if not has_nnaudio: raise ImportError("You must `pip install nnAudio` to use cqt.")
            self.cqt_layer = CQT1992v2(
                sr=self.fs, fmin=1e-4, fmax=5e-2, 
                hop_length=time_ds, bins_per_octave=nv,
                output_format="Complex", trainable=False
            ).to(torch_device)
            dummy = torch.zeros(1, self.N_obs, device=torch_device)
            dummy_cqt = self.cqt_layer(dummy)
            n_freq = dummy_cqt.shape[1]
            n_time = dummy_cqt.shape[2]

        elif repr_type == 'cwt':
            dummy = np.zeros(self.N_obs)
            _, Wx_dummy, *_ = ssq_cwt(dummy, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')
            n_freq = Wx_dummy.shape[0]
            n_time = len(dummy[::time_ds])

        elif repr_type == 'scattering':
            from kymatio.torch import Scattering1D
            scat = Scattering1D(J=J, shape=(self.N_obs,), Q=Q, T=T, max_order=2).to(torch_device)
            meta = scat.meta()
            self.s_indices = np.where(meta['order'] > 0)[0]
            dummy = torch.zeros(1, self.N_obs, device=torch_device)
            scat_dummy = scat(dummy)
            n_freq = len(self.s_indices)
            n_time = scat_dummy.shape[-1]
            
        elif repr_type == 'superlets':
            freqs_short = np.fft.rfftfreq(self.N_obs, self.dt)
            self.freqs_clean = freqs_short[(freqs_short >= 1e-4) & (freqs_short <= 5e-2)]
            n_freq = len(self.freqs_clean)
            n_time = self.N_obs
            
        elif repr_type == 'ssq':
            dummy = np.zeros(self.N_obs)
            Twx = ssq_cwt(dummy, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0]
            n_freq = Twx.shape[0]
            n_time = self.N_obs // time_ds
        
        print(f"[INFO] Initialized Output Grid -> (Channels: {n_ch_out}, Freq: {n_freq}, Time: {n_time})")

        with h5py.File(output_path, 'w') as f:
            f.attrs['dt'] = self.dt
            f.attrs['T_obs'] = self.T_obs
            f.attrs['repr_type'] = repr_type
            f.attrs['channels'] = channels
            f.attrs['param_names'] =["m1", "q", "dist", "t_ref", "inc", "lam", "beta", "psi", "phi_ref", "chi1", "chi2"]
            f.attrs['output_format'] = output_format

            chunk_shape = (1, n_ch_out, n_freq, n_time)
            dset_data = f.create_dataset("data", (n_samples, n_ch_out, n_freq, n_time), dtype='float32', chunks=chunk_shape)
            dset_params = f.create_dataset("parameters", (n_samples, 11), dtype='float32')
            dset_modes = f.create_dataset("modes", (n_samples,), dtype='int32')

            
            for i in tqdm(range(0, n_samples, batch_size)):
                current_bs = min(batch_size, n_samples - i)
                
                batch_params = predefined_population[i:i+current_bs] if predefined_population is not None else None
                mbhb_td_A, mbhb_td_E, mbhb_params = self.get_mbhb_batch(wave_gen, current_bs, predefined_params=batch_params)
                noise_A, noise_E = self.get_noise_batch(current_bs)
                
                mix_td_A = mbhb_td_A + noise_A
                mix_td_E = mbhb_td_E + noise_E

                if self.noise_type == 'psd_whitened':
                    mix_td_A = self.xp.fft.irfft(self.xp.fft.rfft(mix_td_A, axis=-1) / self.sigma_fd_A, n=self.N_obs, axis=-1)
                    mix_td_E = self.xp.fft.irfft(self.xp.fft.rfft(mix_td_E, axis=-1) / self.sigma_fd_E, n=self.N_obs, axis=-1)
                
                if channels == 'XYZ':
                    sq2, sq6, sq3 = self.xp.sqrt(2.0), self.xp.sqrt(6.0), self.xp.sqrt(3.0)
                    mix_X = -(mix_td_A / sq2) + (mix_td_E / sq6)
                    mix_Y = -(2.0 * mix_td_E / sq6)
                    mix_Z =  (mix_td_A / sq2) + (mix_td_E / sq6)
                    mix_td = self.xp.stack([mix_X, mix_Y, mix_Z], axis=1)
                else:
                    mix_td = self.xp.stack([mix_td_A, mix_td_E], axis=1)
                
                if repr_type in['stft', 'scattering', 'cqt']:
                    mix_torch = torch.as_tensor(mix_td, device=torch_device, dtype=torch.float32)
                    mix_torch_flat = mix_torch.view(current_bs * n_ch, self.N_obs)
                
                if repr_type in['superlets', 'ssq', 'cwt']:
                    mix_np = mix_td.get() if self.use_gpu else mix_td
                    mix_np_flat = mix_np.reshape(current_bs * n_ch, self.N_obs)

                if predefined_modes is not None:
                    dset_modes[i:i+current_bs] = predefined_modes[i:i+current_bs]
                else:
                    dset_modes[i:i+current_bs] = 0

                
                ###########################.   REPRESENTATIONS. #################################

                if repr_type == 'stft':
                    Z_mix = torch.stft(mix_torch_flat, n_fft=nperseg, hop_length=hop_length, window=window, 
                                       center=True, return_complex=True, onesided=True)
                    
                    freqs = np.fft.rfftfreq(nperseg, d=self.dt)
                    
                    f_min = 5e-5       # ATTENTION !! IF YOU CHANGE HERE CHANGE ALSO AT LINE 550
                    f_max = 1e-2


                    mask = (freqs >= f_min) & (freqs <= f_max)

                    Z_mix = Z_mix[:, mask, :]
                    
                    # Z_mix_np = Z_mix.cpu().numpy()[:, 1:, :] 
                    # f_lin = np.fft.rfftfreq(nperseg, d=self.dt)[1:]
                    # f_log = np.logspace(np.log10(f_lin[0]), np.log10(f_lin[-1]), num=n_freq)
                    
                    # interp_func = interp1d(f_lin, Z_mix_np, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
                    # Z_mix_log_f = interp_func(f_log).reshape(current_bs, n_ch, n_freq, n_time)  
                    
                    # Z_complex = torch.tensor(Z_mix_log_f, device=torch_device)

                    Z_complex = Z_mix.view(current_bs, n_ch, Z_mix.shape[1], Z_mix.shape[2])

                    if channels == 'AE':
                        Z_A = Z_complex[:, 0, :, :]
                        Z_E = Z_complex[:, 1, :, :]
                        
                        mag_A = torch.abs(Z_A) 
                        mag_E = torch.abs(Z_E) 
                        
                        cross_AE = Z_A * torch.conj(Z_E)
                        
                        cross_AE_norm = cross_AE / (mag_A * mag_E )

                        if apply_log:
                            mag_A = torch.log10(mag_A + 1e-30)   # added 1e-30 because otherwise i could have the log of zero!!!
                            mag_E = torch.log10(mag_E + 1e-30)

                        if output_format == 'mag_phase':
                            phase_diff = torch.angle(cross_AE)
                            data_out = torch.stack([mag_A, mag_E, phase_diff], dim=1).cpu().numpy()
                            
                        elif output_format == 'mag_phase_all':
                            phase_A = torch.angle(Z_A)
                            phase_E = torch.angle(Z_E)
                            phase_diff = torch.angle(cross_AE)

                            # I WOULD UNWRAP PHASE !!!!!!
                            
                            data_out = torch.stack([mag_A, mag_E, phase_A, phase_E, phase_diff], dim=1).cpu().numpy()
                            
                        # elif output_format == 'real_imag':
                        #     cross_re = torch.real(cross_AE_norm)
                        #     cross_im = torch.imag(cross_AE_norm)
                        #     data_out = torch.stack([mag_A, mag_E, cross_re, cross_im], dim=1).cpu().numpy()

                        elif output_format == 'real_imag':

                            re_A = torch.real(Z_A)
                            im_A = torch.imag(Z_A)
                            re_E = torch.real(Z_E)
                            im_E = torch.imag(Z_E)
                            cross_re = torch.real(cross_AE_norm)
                            cross_im = torch.imag(cross_AE_norm)
                            data_out = torch.stack([re_A, im_A, re_E, im_E, cross_re, cross_im], dim=1).cpu().numpy()

                        elif output_format == 'ampl_rel_phase':

                            mag_A = torch.log10(torch.abs(Z_A) )
                            mag_E = torch.log10(torch.abs(Z_E) )

                            cross_AE = Z_A * torch.conj(Z_E)

                            cross_norm = cross_AE / (
                                torch.abs(Z_A) * torch.abs(Z_E) 
                            )

                            cos_dphi = torch.real(cross_norm)
                            sin_dphi = torch.imag(cross_norm)

                            data_out = torch.stack([
                                mag_A,
                                mag_E,
                                cos_dphi,
                                sin_dphi
                            ], dim=1).cpu().numpy()
                            
                        else:
                            data_out = torch.stack([mag_A, mag_E], dim=1).cpu().numpy()
                            
                    else:
                        # IF WE WANT XYZ
                        mag = torch.abs(Z_complex)
                        if apply_log: mag = torch.log10(mag + 1e-30)
                        data_out = mag.cpu().numpy()                  

                
                elif repr_type == 'cqt':   # found this cool implementation of the q transform in nnAudio !!  

                    cqt_raw = self.cqt_layer(mix_torch_flat)
                    cqt_complex = torch.complex(cqt_raw[..., 0], cqt_raw[..., 1])
                    cqt_complex = cqt_complex.view(current_bs, n_ch, n_freq, n_time)

                    # if channels == 'AE':
                    #     CQT_A = cqt_complex[:, 0, :, :]
                    #     CQT_E = cqt_complex[:, 1, :, :]
                        
                    #     cross_AE = CQT_A * torch.conj(CQT_E)
                    #     phase_diff = torch.angle(cross_AE)
                    #     mag_A = torch.abs(CQT_A)
                    #     mag_E = torch.abs(CQT_E)

                    #     if apply_log:
                    #         mag_A = torch.log10(mag_A + 1e-30)
                    #         mag_E = torch.log10(mag_E + 1e-30)

                    #     if output_format == 'mag_phase':
                    #         data_out = torch.stack([mag_A, mag_E, phase_diff], dim=1).cpu().numpy()
                    #     elif output_format == 'real_imag':
                    #         cross_re = torch.real(cross_AE)
                    #         cross_im = torch.imag(cross_AE)
                    #         if apply_log:
                    #             cross_re = torch.sign(cross_re) * torch.log10(1 + torch.abs(cross_re))
                    #             cross_im = torch.sign(cross_im) * torch.log10(1 + torch.abs(cross_im))
                    #         data_out = torch.stack([mag_A, mag_E, cross_re, cross_im], dim=1).cpu().numpy()
                    #     else:
                    #         data_out = torch.stack([mag_A, mag_E], dim=1).cpu().numpy()

                    if channels == 'AE':
                        CQT_A = cqt_complex[:, 0, :, :]
                        CQT_E = cqt_complex[:, 1, :, :]
                        
                        mag_A = torch.abs(CQT_A)
                        mag_E = torch.abs(CQT_E)
                        cross_AE_norm = (CQT_A * torch.conj(CQT_E)) / (mag_A * mag_E )

                        # if apply_log:
                            # mag_A = torch.log10(mag_A)     
                            # mag_E = torch.log10(mag_E)
                            # cross_AE_norm = (CQT_A * torch.conj(CQT_E)) / (mag_A_raw * mag_E_raw + 1e-30)

                        if apply_log:
                            mag_A = torch.log10(mag_A + 1e-30)   # added 1e-30 because otherwise i could have the log of zero!!!
                            mag_E = torch.log10(mag_E + 1e-30)


                        if output_format == 'mag_phase':
                            phase_diff = torch.angle(cross_AE_norm)
                            data_out = torch.stack([mag_A, mag_E, phase_diff], dim=1).cpu().numpy()

                        elif output_format == 'mag_phase_all':
                            phase_A = torch.angle(CQT_A)
                            phase_E = torch.angle(CQT_E)
                            phase_diff = torch.angle(cross_AE_norm)
                            data_out = torch.stack([mag_A, mag_E, phase_A, phase_E, phase_diff], dim=1).cpu().numpy()

                        elif output_format == 'ampl_rel_phase':
                            cos_dphi = torch.real(cross_AE_norm)
                            sin_dphi = torch.imag(cross_AE_norm)
                            data_out = torch.stack([mag_A, mag_E, cos_dphi, sin_dphi], dim=1).cpu().numpy()

                        elif output_format == 'real_imag':

                            re_A = torch.real(CQT_A)
                            im_A = torch.imag(CQT_A)
                            re_E = torch.real(CQT_E)
                            im_E = torch.imag(CQT_E)
                            cross_re = torch.real(cross_AE_norm)
                            cross_im = torch.imag(cross_AE_norm)
                            data_out = torch.stack([re_A, im_A, re_E, im_E, cross_re, cross_im], dim=1).cpu().numpy()
                            # cross_re = torch.real(cross_AE)
                            # cross_im = torch.imag(cross_AE)
                            # data_out = torch.stack([mag_A, mag_E, cross_re, cross_im], dim=1).cpu().numpy()
                        else:
                            data_out = torch.stack([mag_A, mag_E], dim=1).cpu().numpy()

                elif repr_type == 'scattering':
                    S_mix = scat(mix_torch_flat)
                    S_features = S_mix[:, self.s_indices, :]
                    S_features = S_features.view(current_bs, n_ch, S_features.shape[1], S_features.shape[2])
                    data_out = torch.log10(torch.abs(S_features) + 1e-30).cpu().numpy()

                elif repr_type == 'superlets':
                    sl_mix = np.stack([superlets(data=x * 1e22, fs=self.fs, foi=self.freqs_clean, c1=c1, ord=(ord_min, ord_max)) for x in mix_np_flat])
                    sl_mix = sl_mix.reshape(current_bs, n_ch, sl_mix.shape[1], sl_mix.shape[2])
                    data_out = np.log10(sl_mix + 1e-30)

                elif repr_type == 'ssq':
                    try:
                        Twx_mix = ssq_cwt(mix_np_flat * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0]
                    except Exception:
                        Twx_mix = np.stack([ssq_cwt(x * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0] for x in mix_np_flat])
                        
                    Twx_mix = Twx_mix.reshape(current_bs, n_ch, Twx_mix.shape[1], Twx_mix.shape[2])
                    Twx_mix = Twx_mix[:, :, :, ::time_ds][:, :, ::-1, :]
                    
                    mag = np.log10(np.abs(Twx_mix) )
                    if phase_info:
                        data_out = np.concatenate([mag, np.angle(Twx_mix)], axis=1)
                    else:
                        data_out = mag

                elif repr_type == 'cwt':
                    try:
                        _, Wx_mix, *_ = ssq_cwt(mix_np_flat * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')
                    except Exception:
                        Wx_mix = np.stack([ssq_cwt(x * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[1] for x in mix_np_flat])
                        
                    Wx_mix = Wx_mix.reshape(current_bs, n_ch, Wx_mix.shape[1], Wx_mix.shape[2])
                    Wx_mix = Wx_mix[:, :, :, ::time_ds][:, :, ::-1, :]
                    
                    if output_format == 'real_imag':
                        real_part, imag_part = np.real(Wx_mix), np.imag(Wx_mix)
                        if apply_log:
                            real_part = np.sign(real_part) * np.log10(1 + np.abs(real_part))
                            imag_part = np.sign(imag_part) * np.log10(1 + np.abs(imag_part))
                        data_out = np.concatenate([real_part, imag_part], axis=1)
                    elif output_format == 'mag_phase':
                        mag = np.abs(Wx_mix)
                        if apply_log: mag = np.log10(mag + 1e-30)
                        data_out = np.concatenate([mag, np.angle(Wx_mix)], axis=1)
                    else:
                        mag = np.abs(Wx_mix)
                        if apply_log: mag = np.log10(mag + 1e-30)
                        data_out = mag

                dset_data[i:i+current_bs] = data_out
                dset_params[i:i+current_bs] = mbhb_params

                if self.use_gpu:
                    cp.get_default_memory_pool().free_all_blocks()
                    torch.cuda.empty_cache()

                del mbhb_td_A, mbhb_td_E, noise_A, noise_E, mbhb_params
                del mix_td_A, mix_td_E, mix_td
                gc.collect()

                if (i // batch_size) % 10 == 0:
                    f.flush()

def plot_generated_sample(h5_path, out_img="generated_sample.png"):
    
    print(f"\n--- Plotting Sample from {h5_path} ---")
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][0] 
        repr_type = f.attrs['repr_type']
        out_fmt = f.attrs['output_format']
        mode = f['modes'][0] if 'modes' in f else "Unknown"
        
    num_ch = data.shape[0]
    fig, axes = plt.subplots(num_ch, 1, figsize=(10, 3*num_ch))
    if num_ch == 1: axes =[axes]
    
    titles = [f"Channel {i}" for i in range(num_ch)]
    if out_fmt == 'real_imag' and num_ch == 4:
        titles = ["Mag A", "Mag E", "Cross-Spectrum (Real)", "Cross-Spectrum (Imag)"]
    elif out_fmt == 'mag_phase' and num_ch == 3:
        titles = ["Mag A", "Mag E", "Phase Diff"]
    elif out_fmt == 'mag_phase_all' and num_ch == 5:
        titles = ["Mag A", "Mag E", "Absolute Phase A", "Absolute Phase E", "Phase Diff"]
    elif out_fmt == 'amp_rel_phase' and num_ch == 4:
        titles = ["Log Mag A", "Log Mag E", "cos (Phase Diff)", "sin (Phase Diff)"]
    

    for i in range(num_ch):
        cmap = 'twilight' if ('phase' in out_fmt.lower() and i == num_ch-1) else 'viridis'
        if 'Cross-Spectrum' in titles[i]: cmap = 'RdBu'
        if 'cos' in titles[i] or 'sin' in titles[i]: cmap = 'RdBu'
            
        im = axes[i].imshow(data[i], aspect='auto', origin='lower', cmap=cmap)
        axes[i].set_title(titles[i])
        axes[i].set_ylabel("Frequency Bins")
        fig.colorbar(im, ax=axes[i])

    axes[-1].set_xlabel("Time Bins")
    plt.suptitle(f"Method: {repr_type.upper()} | Output Format: {out_fmt} | Exp. Modes: {mode}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    print(f"-> Successfully saved visualization to: {out_img}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--N", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--out", type=str, default="dataset.h5")
    parser.add_argument("--channels", type=str, default="AE", choices=["XYZ", "AE"])
    parser.add_argument("--noise_type", type=str, default="psd_whitened", choices=["gaussian", "psd", "psd_whitened"])
    parser.add_argument("--repr_type", type=str, default="cqt", choices=["stft", "scattering", "superlets", "ssq", "cwt", "cqt"])
    parser.add_argument("--phase_info", action="store_true")
    
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--noverlap", type=int, default=800)
    parser.add_argument("--J", type=int, default=8)
    parser.add_argument("--Q", type=int, default=8)
    parser.add_argument("--T_scat", type=int, default=1)
    parser.add_argument("--c1", type=int, default=8)
    parser.add_argument("--ord_min", type=int, default=1)
    parser.add_argument("--ord_max", type=int, default=5)
    parser.add_argument("--nv", type=int, default=16, help="Bins per octave for CQT/CWT/SSQ")
    parser.add_argument("--ssq_downsample", type=int, default=128, help="Time downsampling for CWT/CQT/SSQ")
    parser.add_argument("--output_format", type=str, default="real_imag", 
                        choices=["real_imag", "mag_phase", "mag_phase_all", "mag", 'ampl_rel_phase'], 
                        help="Output format for representations.")    
    parser.add_argument("--apply_log", action="store_true", default=False)

    # ALBERTO'S POPULATION
    parser.add_argument("--alberto_repo", type=str, default="/sps/lisaf/mbellotti/github/standard-sirens-MBHBs", help="Path to Alberto's cloned repository.")
    parser.add_argument("--alberto_pop", type=str, default="Pop3", choices=["Pop3", "Q3d", "Q3nd"], help="Which population to load.")
    parser.add_argument("--alberto_modes", type=int, default=0, help="Filter by modes (1, 2, or 8). 0 = all.")

    args = parser.parse_args()
    use_gpu = (args.device == "cuda" and use_gpu_available)


    # predefined_pop = None
    # predefined_modes = None
    # if args.alberto_repo != "":
    #     print(f"\n--- Loading Population from Alberto's Repo: {args.alberto_pop} ---")
    #     catalog_params, catalog_modes = load_alberto_population(args.alberto_repo, pop_name=args.alberto_pop, filter_modes=args.alberto_modes)
        
    #     if len(catalog_params) == 0:
    #         print("ERROR: No valid JSON files found. Check your repo path!")
    #         return
        
    #     if args.N > 0:
    #         synthetic_params = generate_astrophysical_prior(catalog_params, args.N)
    #         synthetic_modes = np.zeros(args.N, dtype=int) # 0 means "synthetic KDE sample"
            
    #         # concatenate alberto's pop to the new samples !!!!!!
    #         predefined_pop = np.vstack([synthetic_params, catalog_params])
    #         predefined_modes = np.concatenate([synthetic_modes, catalog_modes])
            
    #         args.N = len(predefined_pop) 
    #         print(f"-> Combined {len(synthetic_params)} KDE samples with {len(catalog_params)} true catalog samples.")
    #         print(f"-> Total events to simulate: {args.N}")
            
    #     else:
    #         # if --N 0, it will generate Alberto's catalog
    #         predefined_pop = catalog_params
    #         predefined_modes = catalog_modes
    #         args.N = len(predefined_pop)
    #         print(f"-> Generating exactly the {args.N} true catalog samples.")


    predefined_pop = None
    predefined_modes = None
    
    if args.alberto_repo != "":
        print(f"\n--- Loading Population from Alberto's Repo: {args.alberto_pop} ---")
        
        # Load the massive catalog for the KDE, and the specific benchmark events for testing
        full_catalog_params, test_params, test_modes = load_alberto_population(args.alberto_repo, pop_name=args.alberto_pop)
        
        if len(full_catalog_params) == 0:
            print("ERROR: No valid JSON files found. Check your repo path!")
            return
        
        if args.N > 0:
            synthetic_params = generate_astrophysical_prior(full_catalog_params, args.N)
            synthetic_modes = np.zeros(args.N, dtype=int) 
            
            if args.N > 100:
                predefined_pop = np.vstack([synthetic_params, test_params])
                predefined_modes = np.concatenate([synthetic_modes, test_modes])
            else:
                predefined_pop = synthetic_params
                predefined_modes = synthetic_modes
            
            args.N = len(predefined_pop) 
            print(f"-> Combined {len(synthetic_params)} KDE samples with {len(predefined_pop) - len(synthetic_params)} true benchmark samples.")
            print(f"-> Total events to simulate: {args.N}")

    gen = DatasetGenerator(dt=10.0, T_obs=7.5*24*3600, T_gen=30*24*3600, noise_type=args.noise_type, use_gpu=use_gpu)
    
    if use_gpu:
        orbits = EqualArmlengthOrbits(use_gpu=True, force_backend="cuda")
        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), response_kwargs=dict(orbits=orbits), force_backend="cuda")
    else:
        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
        
    gen.generate(args.N, wave_gen, args.out, args.batch, 
                 repr_type=args.repr_type, channels=args.channels, 
                 nperseg=args.nperseg, noverlap=args.noverlap,
                 J=args.J, Q=args.Q, T=args.ßT_scat,
                 c1=args.c1, ord_min=args.ord_min, ord_max=args.ord_max, nv=args.nv, phase_info=args.phase_info, 
                 ssq_downsample=args.ssq_downsample, output_format=args.output_format, apply_log=args.apply_log,
                 predefined_population=predefined_pop, predefined_modes=predefined_modes)

    # if args.N < 10 :
    #     plot_generated_sample(args.out, out_img=f"plot_{args.repr_type}_{args.output_format}.png")
    plot_generated_sample(args.out, out_img=f"plot_{args.repr_type}_{args.output_format}.png")

if __name__ == "__main__":
    main()