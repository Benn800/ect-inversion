
import numpy as np
import pandas as pd

def extract_unique_freqs(df):
    return sorted({int(col.split('_')[-1]) for col in df.columns[2:]})

def compute_phase(df):
    new_d = {
    'permeability': df['permeability'],
    'resistivity': df['resistivity'],
    }
    frequencies = extract_unique_freqs(df)
    for f in frequencies:
        h_imag = df[f'H_imag_{f}']
        h_real = df[f'H_real_{f}']
        new_d[f'phase_{f}'] = np.arctan2(h_imag, h_real)
    return pd.DataFrame(new_d)    

def compute_magnitude(df):
    new_d = {
    'permeability': df['permeability'],
    'resistivity': df['resistivity'],
    }
    frequencies = extract_unique_freqs(df)
    for f in frequencies:
        h_imag = df[f'H_imag_{f}']
        h_real = df[f'H_real_{f}']
        new_d[f'magnitude_{f}'] = np.sqrt(h_imag**2 + h_real**2)
    return pd.DataFrame(new_d) 