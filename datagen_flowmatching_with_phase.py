import gc
import fix_cupy
import numpy as np
import h5py
import argparse
import torch
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
            
            # try:
            #     psd_T = get_sensitivity(f_safe, sens_fn="T1TDISens")
            # except Exception:
            #     psd_T = psd_A
            # self.psd_T = cp.asarray(psd_T) if self.use_gpu else psd_T
            # self.sigma_fd_T = self.xp.sqrt(self.psd_T / (4 * self.df))
            
    def get_mbhb_batch(self, wave_gen, batch_size):
        m1 = 10**self.xp.random.uniform(5, 6, batch_size)
        q = self.xp.random.uniform(0.1, 1, batch_size)
        m2 = m1 * q
        # dist = 10**self.xp.random.uniform(0, np.log10(20), batch_size) * 1e9 * PC_SI
        dist = self.xp.random.uniform(self.dist_min, self.dist_max, batch_size)

        t_ref = self.xp.full(batch_size, self.T_gen) - self.T_obs * self.xp.random.uniform(0.1, 0.5, batch_size)
        inc = self.xp.arccos(self.xp.random.uniform(-1, 1, batch_size))
        lam = self.xp.random.uniform(0, 2*np.pi, batch_size)
        beta = self.xp.arcsin(self.xp.random.uniform(-1, 1, batch_size))
        psi = self.xp.random.uniform(0, np.pi, batch_size)   # TODO!! POL ANGLE MAYBE I CAN TRY TO FIX IT!!! to zero? 
        phi_ref = self.xp.random.uniform(0, 2*np.pi, batch_size)
        f_ref = self.xp.zeros(batch_size) 
        zeros = self.xp.zeros(batch_size) 
        modes =[(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        
        freqs_in = cp.asarray(self.freqs_gen) if self.use_gpu else self.freqs_gen

        t_obs_end_yrs = float(self.T_gen) / YRSID_SI

        wave_out = wave_gen(
            m1, m2, zeros, zeros, dist,
            phi_ref, f_ref, inc, lam, beta, psi, t_ref,
            freqs=freqs_in, modes=modes, direct=False, fill=True, 
            squeeze=False, length=self.N_gen,
            t_obs_start=0.0, t_obs_end=t_obs_end_yrs
        )
        
        wave_fd_A = wave_out[:, 0, :] 
        wave_fd_E = wave_out[:, 1, :]
        # wave_fd_T = wave_out[:, 2, :] if wave_out.shape[1] > 2 else self.xp.zeros_like(wave_fd_A)
        
        if self.use_gpu:
            wave_td_A = cp.fft.irfft(wave_fd_A, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            wave_td_E = cp.fft.irfft(wave_fd_E, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            # wave_td_T = cp.fft.irfft(wave_fd_T, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            params = cp.asnumpy(cp.stack([m1, q, dist, t_ref, inc, lam, beta, psi, phi_ref], axis=1))
        else:
            wave_td_A = np.fft.irfft(wave_fd_A, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            wave_td_E = np.fft.irfft(wave_fd_E, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            # wave_td_T = np.fft.irfft(wave_fd_T, n=self.N_gen, axis=-1)[:, -self.N_obs:]
            params = np.stack([m1, q, dist, t_ref, inc, lam, beta, psi, phi_ref], axis=1)
            
        # return wave_td_A, wave_td_E, wave_td_T, params
        return wave_td_A, wave_td_E, params

    def get_noise_batch(self, batch_size):
        if self.noise_type == 'gaussian':  ## old noise !!! not psd
            noise_A = self.xp.random.normal(0, 1e-21, (batch_size, self.N_obs))
            noise_E = self.xp.random.normal(0, 1e-21, (batch_size, self.N_obs))
            noise_T = self.xp.random.normal(0, 1e-21, (batch_size, self.N_obs))
            return noise_A, noise_E, noise_T

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
        # noise_T = gen_psd_noise(self.sigma_fd_T, self.psd_T)

        # return noise_A, noise_E, noise_T
        return noise_A, noise_E

    def generate(self, n_samples, wave_gen, output_path, batch_size, 
                 repr_type='stft', channels='XYZ', nperseg=1024, noverlap=768, 
                 J=8, Q=8, T=1, c1=8, ord_min=1, ord_max=5, nv=16, phase_info=False, ssq_downsample=64, output_format='real_imag', apply_log=False):
                     
        torch_device = "cuda" if self.use_gpu else "cpu"
        
        n_ch = 3 if channels == 'XYZ' else 2


        if repr_type == 'cqt' and channels == 'AE':
            if output_format == 'mag_phase': n_ch_out = 3   #[MagA, MagE, CrossPhase]
            elif output_format == 'real_imag': n_ch_out = 4 # [MagA, MagE, Re(Cross), Im(Cross)]
            else: n_ch_out = 2


        # n_ch_out = n_ch * 2 if repr_type in ['stft', 'ssq'] and phase_info == True else n_ch
        elif repr_type in ['stft', 'cwt', 'cqt']:
            n_ch_out = n_ch * 2 if output_format in['mag_phase', 'real_imag'] else n_ch
        else:
            n_ch_out = n_ch * 2 if phase_info else n_ch
        
        # n_ch_out = n_ch * 2 if repr_type in ['stft', 'ssq'] and phase_info == True else n_ch
        if repr_type in ['stft', 'cwt']:
            n_ch_out = n_ch * 2 if output_format in ['mag_phase', 'real_imag'] else n_ch
        else:
            n_ch_out = n_ch * 2 if phase_info else n_ch
        #FOR SSQ WE NEED TO WDOWNSAMPLE. OTHERWISE TOO BIG...
        time_ds = int(ssq_downsample)


        if repr_type in ['stft']:
            hop_length = nperseg - noverlap
            window = torch.hann_window(nperseg, device=torch_device)
            dummy = torch.zeros(self.N_obs, device=torch_device)
            stft_dummy = torch.stft(dummy, n_fft=nperseg, hop_length=hop_length, window=window, 
                                   center=True, return_complex=True)
            n_freq = 128  
            n_time = stft_dummy.shape[-1]
            print(f"[INFO] LOG-STFT Shape -> (Channels: {n_ch_out} [mag+phase], Freq: {n_freq}, Time: {n_time})")
        
        elif repr_type == 'cwt':
            dummy = np.zeros(self.N_obs)
            _, Wx_dummy, *_ = ssq_cwt(dummy, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')
            n_freq = Wx_dummy.shape[0]
            n_time = len(dummy[::time_ds])
            print(f"[INFO] CWT Shape -> (Channels: {n_ch_out}, Freq: {n_freq}, Time: {n_time})")

        elif repr_type == 'scattering':
            from kymatio.torch import Scattering1D
            scat = Scattering1D(J=J, shape=(self.N_obs,), Q=Q, T=T, max_order=2).to(torch_device)
            meta = scat.meta()
            self.s_indices = np.where(meta['order'] > 0)[0]
            dummy = torch.zeros(1, self.N_obs, device=torch_device)
            scat_dummy = scat(dummy)
            n_freq = len(self.s_indices)
            n_time = scat_dummy.shape[-1]
            print(f"[INFO] SCATTERING Shape -> (Channels: {n_ch_out}, Features(Ord1+2): {n_freq}, Time: {n_time})")
            
        elif repr_type == 'superlets':
            freqs_short = np.fft.rfftfreq(self.N_obs, self.dt)
            self.freqs_clean = freqs_short[(freqs_short >= 1e-4) & (freqs_short <= 5e-2)]
            n_freq = len(self.freqs_clean)
            n_time = self.N_obs
            print(f"[INFO] SUPERLETS Shape -> (Channels: {n_ch_out}, Freq: {n_freq}, Time: {n_time})")

        elif repr_type == 'ssq':
            
            dummy = np.zeros(self.N_obs)
            Twx = ssq_cwt(dummy, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0]
            n_freq = Twx.shape[0]
            #n_time = self.N_obs

            # DOWNSAMPLE ...
            n_time = self.N_obs // time_ds
            print(f"[INFO] SSQ Shape -> (Channels: {n_ch_out} [mag+phase], Freq: {n_freq}, Time: {n_time})")
        else:
            raise ValueError(f"Unknown representation: {repr_type}")
        
        with h5py.File(output_path, 'w') as f:
            f.attrs['dt'] = self.dt
            f.attrs['T_obs'] = self.T_obs
            f.attrs['repr_type'] = repr_type
            f.attrs['channels'] = channels
            f.attrs['param_names'] =["m1", "q", "dist", "t_ref", "inc", "lam", "beta", "psi", "phi_ref"]
            f.attrs['output_format'] = output_format

            chunk_shape = (1, n_ch_out, n_freq, n_time)
            dset_data = f.create_dataset("data", (n_samples, n_ch_out, n_freq, n_time), dtype='float32', chunks=chunk_shape)
            dset_target = f.create_dataset("target", (n_samples, n_ch_out, n_freq, n_time), dtype='float32', chunks=chunk_shape)
            dset_params = f.create_dataset("parameters", (n_samples, 9), dtype='float32')
            # dset_data = f.create_dataset("data", (n_samples, n_ch_out, n_freq, n_time), dtype='float32')
            # dset_target = f.create_dataset("target", (n_samples, n_ch_out, n_freq, n_time), dtype='float32')
            # dset_params = f.create_dataset("parameters", (n_samples, 9), dtype='float32')
            
            for i in tqdm(range(0, n_samples, batch_size)):
                current_bs = min(batch_size, n_samples - i)
                
                # mbhb_td_A, mbhb_td_E, mbhb_td_T, mbhb_params = self.get_mbhb_batch(wave_gen, current_bs)
                # noise_A, noise_E, noise_T = self.get_noise_batch(current_bs)
                mbhb_td_A, mbhb_td_E, mbhb_params = self.get_mbhb_batch(wave_gen, current_bs)
                noise_A, noise_E = self.get_noise_batch(current_bs)
                
                mix_td_A = mbhb_td_A + noise_A
                mix_td_E = mbhb_td_E + noise_E
                # mix_td_T = mbhb_td_T + noise_T

                if self.noise_type == 'psd_whitened':
                    mix_td_A = self.xp.fft.irfft(self.xp.fft.rfft(mix_td_A, axis=-1) / self.sigma_fd_A, n=self.N_obs, axis=-1)
                    mix_td_E = self.xp.fft.irfft(self.xp.fft.rfft(mix_td_E, axis=-1) / self.sigma_fd_E, n=self.N_obs, axis=-1)
                    # mix_td_T = self.xp.fft.irfft(self.xp.fft.rfft(mix_td_T, axis=-1) / self.sigma_fd_T, n=self.N_obs, axis=-1)
                
                if channels == 'XYZ':
                    sq2, sq6, sq3 = self.xp.sqrt(2.0), self.xp.sqrt(6.0), self.xp.sqrt(3.0)
                    mix_X = -(mix_td_A / sq2) + (mix_td_E / sq6) + (mix_td_T / sq3)
                    mix_Y = -(2.0 * mix_td_E / sq6) + (mix_td_T / sq3)
                    mix_Z =  (mix_td_A / sq2) + (mix_td_E / sq6) + (mix_td_T / sq3)
                    mix_td = self.xp.stack([mix_X, mix_Y, mix_Z], axis=1)
                else:
                    mix_td = self.xp.stack([mix_td_A, mix_td_E], axis=1)
                
                if repr_type in ['stft', 'scattering']:
                    mix_torch = torch.as_tensor(mix_td, device=torch_device, dtype=torch.float32)
                    mix_torch_flat = mix_torch.view(current_bs * n_ch, self.N_obs)
                
                if repr_type in ['superlets', 'ssq', 'cwt']:
                    mix_np = mix_td.get() if self.use_gpu else mix_td
                    mix_np_flat = mix_np.reshape(current_bs * n_ch, self.N_obs)


                # if repr_type == 'stft':
                #     Z_mix = torch.stft(mix_torch_flat, n_fft=nperseg, hop_length=hop_length, window=window, 
                #                        center=True, return_complex=True, onesided=True)
                    
                #     Z_mix_np = Z_mix.cpu().numpy()[:, 1:, :] 
                #     f_lin = np.fft.rfftfreq(nperseg, d=self.dt)[1:]
                    
                #     f_log = np.logspace(np.log10(f_lin[0]), np.log10(f_lin[-1]), num=n_freq)
                #     #interp_func = interp1d(f_lin, Z_mix_np, axis=1, kind='linear')
                #     interp_func = interp1d(f_lin, Z_mix_np, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
                #     Z_mix_log = interp_func(f_log)
                    
                #     Z_mix_log = Z_mix_log.reshape(current_bs, n_ch, n_freq, n_time)

                #     mag = np.log10(np.abs(Z_mix_log) + 1e-30)
                    
                #     if phase_info:
                #         phase = np.angle(Z_mix_log)
                #         data_out = np.concatenate([mag, phase], axis=1) 
                #     else:
                #         data_out = mag

                if repr_type == 'stft':
                    Z_mix = torch.stft(mix_torch_flat, n_fft=nperseg, hop_length=hop_length, window=window, 
                                       center=True, return_complex=True, onesided=True)
                    
                    Z_mix_np = Z_mix.cpu().numpy()[:, 1:, :] 
                    f_lin = np.fft.rfftfreq(nperseg, d=self.dt)[1:]
                    f_log = np.logspace(np.log10(f_lin[0]), np.log10(f_lin[-1]), num=n_freq)
                    
                    mag_lin = np.abs(Z_mix_np)
                    
                    phase_lin = np.unwrap(np.angle(Z_mix_np), axis=1) 
                    
                    # interp_mag = interp1d(f_lin, mag_lin, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
                    # interp_phase = interp1d(f_lin, phase_lin, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
                    interp_func = interp1d(f_lin, Z_mix_np, axis=1, kind='linear', bounds_error=False, fill_value="extrapolate")
                    Z_mix_log_f = interp_func(f_log).reshape(current_bs, n_ch, n_freq, n_time)  

                    if output_format == 'real_imag':
                        data_out = np.concatenate([np.real(Z_mix_log_f), np.imag(Z_mix_log_f)], axis=1)
                    elif output_format == 'mag_phase':
                        mag = np.abs(Z_mix_log_f)
                        if apply_log: mag = np.log10(mag + 1e-30)
                        phase = np.angle(Z_mix_log_f)
                        data_out = np.concatenate([mag, phase], axis=1)
                    else: # 'mag'
                        mag = np.abs(Z_mix_log_f)
                        if apply_log: mag = np.log10(mag + 1e-30)
                        data_out = mag                    


                    # mag_log = interp_mag(f_log).reshape(current_bs, n_ch, n_freq, n_time)
                    # phase_log = interp_phase(f_log).reshape(current_bs, n_ch, n_freq, n_time)
                    
                    # mag_out = np.log10(mag_log + 1e-30)
                    
                    # if phase_info:
                    #     data_out = np.concatenate([mag_out, phase_log], axis=1) 
                    # else:
                    #     data_out = mag_out        


                elif repr_type == 'scattering':
                    S_mix = scat(mix_torch_flat)
                    # Use both Order 1 and Order 2 coefficients
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

                    Twx_mix = Twx_mix[:, :, :, ::time_ds]
                    Twx_mix = Twx_mix[:, :, ::-1, :]
                    
                    mag = np.log10(np.abs(Twx_mix) )
                    if phase_info:
                        phase = np.angle(Twx_mix)
                        data_out = np.concatenate([mag, phase], axis=1)
                    else:
                        data_out = mag

                elif repr_type == 'cwt':
                    try:
                        _, Wx_mix, *_ = ssq_cwt(mix_np_flat * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')
                    except Exception:
                        Wx_mix = np.stack([ssq_cwt(x * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[1] for x in mix_np_flat])
                        
                    Wx_mix = Wx_mix.reshape(current_bs, n_ch, Wx_mix.shape[1], Wx_mix.shape[2])
                    Wx_mix = Wx_mix[:, :, :, ::time_ds]
                    Wx_mix = Wx_mix[:, :, ::-1, :]

                    
                    if output_format == 'real_imag':
                        real_part = np.real(Wx_mix)
                        imag_part = np.imag(Wx_mix)
                        
                        if apply_log:
                            print('applying log')
                            # ! NOT SURE IT'S THE WAY I SHOULD APPLY LOG SCALE.. BUT I HAVE POSITIVE AND NEGATIVE NUMBERS... 
                            # ! WITHOUT LOG CANNOT SEE HIGHER MODES... BY EYE... WILL THE NETWORK HAVE THE SAME PROBLEM?
                            real_part = np.sign(real_part) * np.log10(1 + np.abs(real_part))
                            imag_part = np.sign(imag_part) * np.log10(1 + np.abs(imag_part))
                            
                        data_out = np.concatenate([real_part, imag_part], axis=1)


                        # data_out = np.concatenate([np.real(Wx_mix), np.imag(Wx_mix)], axis=1)
                    elif output_format == 'mag_phase':
                        mag = np.abs(Wx_mix)
                        if apply_log: 
                            mag = np.log10(mag + 1e-30)
                        phase = np.angle(Wx_mix)
                        data_out = np.concatenate([mag, phase], axis=1)
                    else:
                        mag = np.abs(Wx_mix)
                        if apply_log: 
                            mag = np.log10(mag + 1e-30)
                        data_out = mag



                dset_data[i:i+current_bs] = data_out
                dset_target[i:i+current_bs] = data_out # If you wanted to denoise, you'd calculate clean phase too. But for flow matching, this is fine.
                dset_params[i:i+current_bs] = mbhb_params

                if self.use_gpu:
                    cp.get_default_memory_pool().free_all_blocks()
                    torch.cuda.empty_cache()

                # del mbhb_td_A, mbhb_td_E, mbhb_td_T, noise_A, noise_E, noise_T, mbhb_params
                del mbhb_td_A, mbhb_td_E, noise_A, noise_E, mbhb_params
                # del mix_td_A, mix_td_E, mix_td, mix_np_flat, mix_torch_flat
                del mix_td_A, mix_td_E, mix_td
                if repr_type == 'ssq':
                    del Twx_mix, data_out, mag
                    # if phase_info:
                    #     del phase
                elif repr_type == 'stft':
                    # del Z_mix, Z_mix_np, Z_mix_log, data_out, mag
                    del Z_mix, Z_mix_np, Z_mix_log_f, data_out, mix_torch_flat

                elif repr_type == 'cwt':
                    del Wx_mix, data_out


                gc.collect()

                if (i // batch_size) % 10 == 0:
                    f.flush()

def main():#
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", type=str, default="dataset.h5")
    parser.add_argument("--channels", type=str, default="XYZ", choices=["XYZ", "AE"])
    parser.add_argument("--noise_type", type=str, default="psd_whitened", choices=["gaussian", "psd", "psd_whitened"])
    parser.add_argument("--repr_type", type=str, default="stft", choices=["stft", "scattering", "superlets", "ssq", "cwt"])
    parser.add_argument("--phase_info", action="store_true", help="Include phase information in the representation (only applicable for 'stft' and 'ssq'). If True, output channels will be doubled to include both magnitude and phase.")
    
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--noverlap", type=int, default=992)

    parser.add_argument("--J", type=int, default=8)
    parser.add_argument("--Q", type=int, default=8)
    parser.add_argument("--T_scat", type=int, default=1)
    parser.add_argument("--c1", type=int, default=8)
    parser.add_argument("--ord_min", type=int, default=1)
    parser.add_argument("--ord_max", type=int, default=5)
    parser.add_argument("--nv", type=int, default=16)
    parser.add_argument("--ssq_downsample", type=int, default=64)
    parser.add_argument("--output_format", type=str, default="real_imag", choices=["real_imag", "mag_phase", "mag"], help="Output format for STFT representation. 'real_imag' keeps real and imaginary parts as separate channels, 'mag_phase' outputs magnitude and phase as separate channels, and 'mag' outputs only the magnitude.")
    parser.add_argument("--apply_log", action="store_true", help="Apply logarithmic scaling to the magnitude of the STFT representation. Only applicable if output_format is 'mag' or 'mag_phase'.")

    args = parser.parse_args()
    
    use_gpu = (args.device == "cuda" and use_gpu_available)
    
    gen = DatasetGenerator(dt=10.0, T_obs=7.5*24*3600, T_gen=30*24*3600, noise_type=args.noise_type, use_gpu=use_gpu)
    
    if use_gpu:
        orbits = EqualArmlengthOrbits(use_gpu=True, force_backend="cuda")
        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), response_kwargs=dict(orbits=orbits), force_backend="cuda")
    else:
        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
        
    gen.generate(args.N, wave_gen, args.out, args.batch, 
                 repr_type=args.repr_type, channels=args.channels, 
                 nperseg=args.nperseg, noverlap=args.noverlap,
                 J=args.J, Q=args.Q, T=args.T_scat,
                 c1=args.c1, ord_min=args.ord_min, ord_max=args.ord_max, nv=args.nv, phase_info=args.phase_info, ssq_downsample=args.ssq_downsample, output_format=args.output_format, apply_log=args.apply_log)

if __name__ == "__main__":
    main()