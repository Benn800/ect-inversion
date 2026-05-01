"""
data.py functions-
load_raw() - loads em_sensor_data.xlsx
rename_columns() - renames the columns for better readability
compute_inductance() - calculates the difference between the real and imag values of inductance for coil 2 and coil 3
reshape_wide() - reshapes data to 55x16 format 
((unique combinations of 11 restivity and 5 permeability values) x (real and imag I values for 8 frequencies))
"""
import pandas as pd
from pathlib import Path

def load_raw():
    # Resolve path from this file so loading does not depend on cwd.
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "em_sensor_data.xlsx"
    df = pd.read_excel(data_path, header=4)
    return df

def rename_columns(df):

    df.columns = ['permeability', 'resistivity', 'frequency',
                'coil2_real', 'coil2_imag',
                'coil3_real', 'coil3_imag']
    return df
    
def compute_inductance(df):
    # Differential mutual inductance: H = coil2 - coil3
    # Both the real and imaginary parts are differenced independently.
    df['H_real'] = df['coil2_real'] - df['coil3_real']
    df['H_imag'] = df['coil2_imag'] - df['coil3_imag']

    # COMSOL coil orientation can produce an overall sign flip relative to
    # the physical measurement convention. If the majority of H_real values
    # are negative, negate the whole column so downstream plots are intuitive.
    if df['H_real'].mean() < 0:
        df['H_real'] = -df['H_real']
    if df['H_imag'].mean() < 0:
        df['H_imag'] = -df['H_imag']
    df = df.drop(labels=['coil2_real', 'coil2_imag', 'coil3_real', 'coil3_imag'], axis=1)    
    return df    


def reshape_wide(df):
    df = pd.pivot_table(df, index=['permeability', 'resistivity'], 
                        columns='frequency', values=['H_real', 'H_imag'])
    df = df.reset_index()
    df.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] 
                  for col in df.columns]
    return df

def load_data():
    df = load_raw()
    df = rename_columns(df)
    df = compute_inductance(df)
    df = reshape_wide(df)
    return df    