import fix_cupy
import numpy as np
import h5py
import argparse
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.utils.constants import PC_SI
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity
from gbgpu.gbgpu import GBGPU
from ssqueezepy import ssq_cwt
from superlet import superlets

try:
    import cupy as cp
    use_gpu_available = True
except ImportError:
    use_gpu_available = False

class DatasetGenerator:
    def __init__(self, dt=10.0, T_obs=172800.0, T_gen=2592000.0, gb_count=5, noise_type='psd_whitened', use_gpu=False):
        self.dt = dt
        self.fs = 1.0 / dt
        self.T_obs = T_obs
        self.df = 1/T_obs
        self.N_obs = int(T_obs / dt)
        self.T_gen = T_gen
        self.N_gen = int(T_gen / dt)
        self.gb_count = gb_count
        self.noise_type = noise_type
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        self.freqs_gen = np.fft.rfftfreq(self.N_gen, dt)
        self.gb = GBGPU(force_backend="cuda" if use_gpu else "cpu")

        if self.noise_type in['psd', 'psd_whitened']:
            freqs_short = np.fft.rfftfreq(self.N_obs, self.dt)
            f_safe = np.clip(freqs_short, 1e-5, None)
            psd_A = get_sensitivity(f_safe, sens_fn="A1TDISens")
            self.psd_A = cp.asarray(psd_A) if self.use_gpu else psd_A
            self.sigma_fd = self.xp.sqrt(self.psd_A / (4 * self.df))
            self.whitening_factor = 1/ self.sigma_fd
            
    def get_mbhb_batch(self, wave_gen, batch_size):
        m1 = 10**np.random.uniform(5, 7, batch_size)
        q = np.random.uniform(0.5, 1, batch_size)
        m2 = m1 * q
        dist = 10**np.random.uniform(0, np.log10(20), batch_size) * 1e9 * PC_SI
        t_ref = np.full(batch_size, self.T_gen) - self.T_obs * np.random.uniform(0.1, 0.6, batch_size)
        inc = np.arccos(np.random.uniform(-1, 1, batch_size))
        lam = np.random.uniform(0, 2*np.pi, batch_size)
        beta = np.arcsin(np.random.uniform(-1, 1, batch_size))
        psi = np.random.uniform(0, np.pi, batch_size)
        phi_ref = np.random.uniform(0, 2*np.pi, batch_size)
        f_ref = np.zeros(batch_size) 
        zeros = np.zeros(batch_size)
        modes =[(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        
        freqs_in = cp.asarray(self.freqs_gen) if self.use_gpu else self.freqs_gen

        wave_out = wave_gen(
            m1, m2, zeros, zeros, dist,
            phi_ref, f_ref, inc, lam, beta, psi, t_ref,
            freqs=freqs_in, modes=modes, direct=False, fill=True, 
            squeeze=False, length=self.N_gen,
            t_obs_start=0.0, t_obs_end=float(self.T_gen)
        )
        
        wave_fd = wave_out[:, 0, :] 
        
        if self.use_gpu:
            wave_td_long = cp.fft.irfft(wave_fd, n=self.N_gen, axis=-1)
        else:
            wave_td_long = np.fft.irfft(wave_fd, n=self.N_gen, axis=-1)
            
        params = np.stack([m1, q, dist, t_ref, inc, lam, beta, psi, phi_ref], axis=1)
            
        return wave_td_long[:, -self.N_obs:], params

    def get_gb_batch(self, batch_size):
        total_gbs = batch_size * self.gb_count
        params = np.zeros((9, total_gbs))
        params[0] = 10**np.random.uniform(-21, -20, total_gbs)
        params[1] = 10**np.random.uniform(np.log10(1e-3), np.log10(1e-2), total_gbs)
        params[2] = 10**np.random.uniform(-18, -16, total_gbs)
        params[3] = 0.0
        params[4] = np.random.uniform(0, 2*np.pi, total_gbs)
        params[5] = np.arccos(np.random.uniform(-1, 1, total_gbs))
        params[6] = np.random.uniform(0, np.pi, total_gbs)
        params[7] = np.random.uniform(0, 2*np.pi, total_gbs)
        params[8] = np.arcsin(np.random.uniform(-1, 1, total_gbs))
        
        params_xp = cp.asarray(params) if self.use_gpu else params
        group_index = self.xp.repeat(self.xp.arange(batch_size, dtype=self.xp.int32), self.gb_count)
        templates = self.xp.zeros((batch_size, 2, self.N_obs // 2 + 1), dtype=self.xp.complex128)
        
        self.gb.generate_global_template(
            params_xp.T, group_index, templates,
            T=self.T_obs, dt=self.dt, N=self.N_obs
        )
        
        if self.use_gpu:
            return cp.fft.irfft(templates[:, 0, :], n=self.N_obs, axis=-1)
        else:
            return np.fft.irfft(templates[:, 0, :], n=self.N_obs, axis=-1)

    def generate(self, n_samples, wave_gen, output_path, batch_size, 
                 repr_type='stft', nperseg=1024, noverlap=768, 
                 J=8, Q=8, T=1, c1=8, ord_min=1, ord_max=5, nv=16):
                     
        torch_device = "cuda" if self.use_gpu else "cpu"

        if repr_type == 'stft':
            hop_length = nperseg - noverlap
            window = torch.hann_window(nperseg, device=torch_device)
            dummy = torch.zeros(self.N_obs, device=torch_device)
            stft_dummy = torch.stft(dummy, n_fft=nperseg, hop_length=hop_length, window=window, 
                                   center=True, return_complex=True)
            n_freq, n_time = stft_dummy.shape
            
        elif repr_type == 'scattering':
            try:
                from kymatio.torch import Scattering1D
            except ImportError:
                raise ImportError("Please install kymatio to use the scattering representation.")
            
            scat = Scattering1D(J=J, shape=(self.N_obs,), Q=Q, T=T, max_order=2).to(torch_device)
            meta = scat.meta()
            self.s1_indices = np.where(meta['order'] == 1)[0]
            dummy = torch.zeros(1, self.N_obs, device=torch_device)
            scat_dummy = scat(dummy)
            n_freq = len(self.s1_indices)
            n_time = scat_dummy.shape[-1]
            
        elif repr_type == 'superlets':
            freqs_short = np.fft.rfftfreq(self.N_obs, self.dt)
            self.freqs_clean = freqs_short[(freqs_short >= 1e-4) & (freqs_short <= 5e-2)]
            n_freq = len(self.freqs_clean)
            n_time = self.N_obs
            
        elif repr_type == 'ssq':
            dummy = np.zeros(self.N_obs)
            # Fetch only the first element to avoid unpacking errors
            Twx = ssq_cwt(dummy, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0]
            n_freq = Twx.shape[0]
            n_time = self.N_obs
            
        else:
            raise ValueError(f"Unknown representation: {repr_type}")
        
        with h5py.File(output_path, 'w') as f:
            f.attrs['dt'] = self.dt
            f.attrs['T_obs'] = self.T_obs
            f.attrs['repr_type'] = repr_type
            
            if repr_type == 'stft':
                f.attrs['nperseg'] = nperseg
                f.attrs['noverlap'] = noverlap
            elif repr_type == 'scattering':
                f.attrs['J'] = J
                f.attrs['Q'] = Q
                f.attrs['T_scat'] = T
            elif repr_type == 'superlets':
                f.attrs['c1'] = c1
                f.attrs['ord_min'] = ord_min
                f.attrs['ord_max'] = ord_max
            elif repr_type == 'ssq':
                f.attrs['nv'] = nv
            
            dset_in = f.create_dataset("data", (n_samples, 1, n_freq, n_time), dtype='float32')
            dset_target = f.create_dataset("target", (n_samples, 1, n_freq, n_time), dtype='float32')
            dset_params = f.create_dataset("parameters", (n_samples, 9), dtype='float32')
            
            for i in tqdm(range(0, n_samples, batch_size)):
                current_bs = min(batch_size, n_samples - i)
                
                mbhb_td, mbhb_params = self.get_mbhb_batch(wave_gen, current_bs)
                gb_td = self.get_gb_batch(current_bs)
                
                if self.noise_type == 'gaussian':
                    noise_td = self.xp.random.normal(0, 1e-21, mbhb_td.shape)
                else:
                    noise_real = self.xp.random.normal(0, self.sigma_fd, (current_bs, len(self.sigma_fd)))
                    noise_imag = self.xp.random.normal(0, self.sigma_fd, (current_bs, len(self.sigma_fd)))
                    noise_imag[:, 0] = 0.0
                    noise_real[:, 0] = self.xp.random.normal(0, self.xp.sqrt(self.N_obs / (2 * self.dt) * self.psd_A[0]), current_bs)
                    if self.N_obs % 2 == 0:
                        noise_imag[:, -1] = 0.0
                        noise_real[:, -1] = self.xp.random.normal(0, self.xp.sqrt(self.N_obs / (2 * self.dt) * self.psd_A[-1]), current_bs)
                    noise_fd = noise_real + 1j * noise_imag
                    noise_td = self.xp.fft.irfft(noise_fd, n=self.N_obs, axis=-1)

                mix_td = mbhb_td + gb_td + noise_td
                target_td = mbhb_td + noise_td

                if self.noise_type == 'psd_whitened':
                    mix_fd = self.xp.fft.rfft(mix_td, axis=-1) * self.whitening_factor
                    mix_td = self.xp.fft.irfft(mix_fd, n=self.N_obs, axis=-1)
                    target_fd = self.xp.fft.rfft(target_td, axis=-1) * self.whitening_factor
                    target_td = self.xp.fft.irfft(target_fd, n=self.N_obs, axis=-1)
                
                if repr_type in ['stft', 'scattering']:
                    mix_torch = torch.as_tensor(mix_td, device=torch_device, dtype=torch.float32)
                    target_torch = torch.as_tensor(target_td, device=torch_device, dtype=torch.float32)
                
                if repr_type in ['superlets', 'ssq']:
                    if self.use_gpu:
                        mix_np = mix_td.get()
                        target_np = target_td.get()
                    else:
                        mix_np = mix_td
                        target_np = target_td

                if repr_type == 'stft':
                    Z_mix = torch.stft(mix_torch, n_fft=nperseg, hop_length=hop_length, window=window, 
                                       center=True, return_complex=True, onesided=True)
                    Z_target = torch.stft(target_torch, n_fft=nperseg, hop_length=hop_length, window=window, 
                                          center=True, return_complex=True, onesided=True)
                    
                    dset_in[i:i+current_bs, 0] = torch.log10(torch.abs(Z_mix) + 1e-30).cpu().numpy()
                    dset_target[i:i+current_bs, 0] = torch.log10(torch.abs(Z_target) + 1e-30).cpu().numpy()
                
                elif repr_type == 'scattering':
                    S_mix = scat(mix_torch)
                    S_target = scat(target_torch)
                    
                    S1_mix = S_mix[:, self.s1_indices, :]
                    S1_target = S_target[:, self.s1_indices, :]
                    
                    dset_in[i:i+current_bs, 0] = torch.log10(torch.abs(S1_mix) + 1e-30).cpu().numpy()
                    dset_target[i:i+current_bs, 0] = torch.log10(torch.abs(S1_target) + 1e-30).cpu().numpy()

                # elif repr_type == 'superlets':
                #     sl_mix = np.stack([superlets(data=x, fs=self.fs, foi=self.freqs_clean, c1=c1, ord=(ord_min, ord_max)) for x in mix_np])
                #     sl_target = np.stack([superlets(data=x, fs=self.fs, foi=self.freqs_clean, c1=c1, ord=(ord_min, ord_max)) for x in target_np])
                    
                #     dset_in[i:i+current_bs, 0] = np.log10(sl_mix + 1e-30)
                #     dset_target[i:i+current_bs, 0] = np.log10(sl_target + 1e-30)

                elif repr_type == 'superlets':
                    # Multiply by 1e22 to avoid float32 underflow and eclipsing the 1e-30 floor
                    sl_mix = np.stack([superlets(data=x * 1e22, fs=self.fs, foi=self.freqs_clean, c1=c1, ord=(ord_min, ord_max)) for x in mix_np])
                    sl_target = np.stack([superlets(data=x * 1e22, fs=self.fs, foi=self.freqs_clean, c1=c1, ord=(ord_min, ord_max)) for x in target_np])
                    
                    dset_in[i:i+current_bs, 0] = np.log10(sl_mix + 1e-30)
                    dset_target[i:i+current_bs, 0] = np.log10(sl_target + 1e-30)

                elif repr_type == 'ssq':
                    try:
                        Twx_mix = ssq_cwt(mix_np * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0]
                        Twx_target = ssq_cwt(target_np * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0]
                    except Exception:
                        Twx_mix = np.stack([ssq_cwt(x * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0] for x in mix_np])
                        Twx_target = np.stack([ssq_cwt(x * 1e22, fs=self.fs, wavelet='morlet', scales='log', nv=nv, padtype='zero')[0] for x in target_np])
                        
                    dset_in[i:i+current_bs, 0] = np.log10(np.abs(Twx_mix) + 1e-3)
                    dset_target[i:i+current_bs, 0] = np.log10(np.abs(Twx_target) + 1e-3)

                dset_params[i:i+current_bs] = mbhb_params

                del mbhb_td, gb_td, mix_td, target_td
                if repr_type in ['stft', 'scattering']:
                    del mix_torch, target_torch
                if repr_type == 'stft':
                    del Z_mix, Z_target
                elif repr_type == 'scattering':
                    del S_mix, S_target, S1_mix, S1_target
                elif repr_type == 'superlets':
                    del sl_mix, sl_target, mix_np, target_np
                elif repr_type == 'ssq':
                    del Twx_mix, Twx_target, mix_np, target_np
                
                if self.use_gpu:
                    cp.get_default_memory_pool().free_all_blocks()
                    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--out", type=str, default="dataset.h5")
    parser.add_argument("--gb_count", type=int, default=5)
    parser.add_argument("--noise_type", type=str, default="psd_whitened", choices=["gaussian", "psd", "psd_whitened"])
    parser.add_argument("--repr_type", type=str, default="stft", choices=["stft", "scattering", "superlets", "ssq"])
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--noverlap", type=int, default=768)
    parser.add_argument("--J", type=int, default=8)
    parser.add_argument("--Q", type=int, default=8)
    parser.add_argument("--T_scat", type=int, default=1)
    parser.add_argument("--c1", type=int, default=8)
    parser.add_argument("--ord_min", type=int, default=1)
    parser.add_argument("--ord_max", type=int, default=5)
    parser.add_argument("--nv", type=int, default=16)
    args = parser.parse_args()
    
    use_gpu = (args.device == "cuda" and use_gpu_available)
    
    gen = DatasetGenerator(dt=10.0, T_obs=48*3600, T_gen=30*24*3600, gb_count=args.gb_count, noise_type=args.noise_type, use_gpu=use_gpu)
    
    if use_gpu:
        orbits = EqualArmlengthOrbits(use_gpu=True, force_backend="cuda")
        wave_gen = BBHWaveformFD(
            amp_phase_kwargs=dict(run_phenomd=False),
            response_kwargs=dict(orbits=orbits),
            force_backend="cuda"
        )
    else:
        wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
        
    gen.generate(args.N, wave_gen, args.out, args.batch, 
                 repr_type=args.repr_type, nperseg=args.nperseg, noverlap=args.noverlap,
                 J=args.J, Q=args.Q, T=args.T_scat,
                 c1=args.c1, ord_min=args.ord_min, ord_max=args.ord_max, nv=args.nv)

if __name__ == "__main__":
    main()