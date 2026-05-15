import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

def plot_test_dataset(h5_path, num_samples_to_plot=4):
    print(f"Opening {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        n_samples, n_ch, n_freq, n_time = f['data'].shape
        print(f"Data shape: {f['data'].shape}")
        
        plot_count = min(n_samples, num_samples_to_plot)
        
        data = f['data'][:plot_count]
        params = f['parameters'][:plot_count]
        
        repr_type = f.attrs.get('repr_type', 'stft')
        if isinstance(repr_type, bytes): repr_type = repr_type.decode('utf-8')
            
        channels_attr = f.attrs.get('channels', 'AE')
        if isinstance(channels_attr, bytes): channels_attr = channels_attr.decode('utf-8')
            
        output_format = f.attrs.get('output_format', 'mag')
        if isinstance(output_format, bytes): output_format = output_format.decode('utf-8')
        
        base_channels = list(channels_attr)
        ch_labels =[]
        
        if repr_type in ['stft', 'ssq', 'cwt']:
            has_extra = (n_ch == len(base_channels) * 2)
            for i in range(n_ch):
                if has_extra:
                    if i < len(base_channels):
                        prefix = "Real" if output_format == 'real_imag' else "Mag"
                        ch_labels.append(f"{prefix} {base_channels[i]}")
                    else:
                        prefix = "Imag" if output_format == 'real_imag' else "Phase"
                        ch_labels.append(f"{prefix} {base_channels[i - len(base_channels)]}")
                else:
                    ch_labels.append(f"Mag {base_channels[i]}")
        elif repr_type == 'scattering':
            for i in range(n_ch):
                ch_labels.append(f"Scattering {base_channels[i]}")

        fig, axes = plt.subplots(plot_count, n_ch, figsize=(6 * n_ch, 4 * plot_count))
        
        if plot_count == 1: axes = np.expand_dims(axes, axis=0)
        if n_ch == 1: axes = np.expand_dims(axes, axis=1)

        for i in range(plot_count):
            t_ref = params[i, 3] 
            
            for c in range(n_ch):
                ax = axes[i, c]
                img = data[i, c]
                
                is_phase = "Phase" in ch_labels[c]
                is_real_imag = "Real" in ch_labels[c] or "Imag" in ch_labels[c]
                
                if is_phase:
                    cmap = 'twilight' 
                    vmin, vmax = -np.pi, np.pi
                elif is_real_imag:
                    cmap = 'RdBu_r'
                    vmax = np.percentile(np.abs(img), 99)
                    vmin = -vmax
                else:
                    cmap = 'viridis'
                    vmin, vmax = np.percentile(img, [1, 99])
                
                im = ax.imshow(img, aspect='auto', origin='lower', 
                               cmap=cmap, vmin=vmin, vmax=vmax)
                
                ax.set_title(f"Sample {i} | {ch_labels[c]}\nt_ref = {t_ref:.1f} s")
                ax.set_xlabel("Time Bins")
                
                if c == 0:
                    ylabel = "Scattering Paths (Ord 1+2)" if repr_type == 'scattering' else "Frequency Bins"
                    ax.set_ylabel(ylabel)
                
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if is_phase:
                    cbar.set_ticks([-np.pi, 0, np.pi])
                    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])

        plt.tight_layout()
        filename = os.path.basename(h5_path).replace('.h5', '')
        output_file = f"plot_{filename}.png"
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved successfully to {output_file}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="test_cwt.h5", help="Path to HDF5 dataset to plot")
    parser.add_argument("--n", type=int, default=4, help="Number of samples to plot")
    args = parser.parse_args()
    
    plot_test_dataset(args.file, args.n)