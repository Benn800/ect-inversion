"""
robustness.py - shared helpers for the liftoff and Gaussian-noise robustness studies.

Both studies compare the full-spectrum dense models against the ZCF baseline, and the
comparison is only meaningful if both methods consume the *same* corrupted data. Every
helper needed by both paths lives here so there is a single definition of each step:

load_liftoff_sweep()   - reads the COMSOL liftoff export into the project's long form
perturb_inductance()   - Gaussian noise on the raw coil inductances, BEFORE differencing
to_wide_features()     - long -> the 16-column H vector, in the training column order
zcf() / zcf_table()    - zero-crossing frequency extraction (moved from 01_zcf.ipynb)
predict_mu_given_rho() - conditional ZCF inversion (moved from 01_zcf.ipynb)
predict_rho_given_mu()

The four ZCF functions are the published baseline verbatim; 01_zcf.ipynb imports them
from here rather than redefining them, so the two studies cannot drift apart.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.data import rename_columns, compute_differential

# The four raw inductance channels, in the order rename_columns produces them.
# Noise is applied to these, never to the differential.
COIL_COLS = ['coil2_real', 'coil2_imag', 'coil3_real', 'coil3_imag']

# Which coil each channel belongs to - the noise scale is set per coil, from the
# complex modulus, so a channel passing through zero is not left noise-free.
COIL_OF = ['coil2', 'coil2', 'coil3', 'coil3']


def load_liftoff_sweep(file_name='liftoff_sweep.csv'):
    """data/raw/liftoff_sweep.csv -> long form [liftoff, frequency, coil2_*, coil3_*].

    The file is a raw COMSOL export: four '%' comment lines, then a '% '-prefixed
    header. Columns are overwritten positionally, the same convention (and the same
    physical ordering) as src.data.rename_columns.

    Carries no permeability/resistivity columns - the material is recovered by
    matching the 40 mm row against the dense grid in 01_liftoff.ipynb.
    """
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / 'data' / 'raw' / file_name, skiprows=4)
    df.columns = ['liftoff', 'frequency',
                  'coil2_real', 'coil2_imag',
                  'coil3_real', 'coil3_imag']
    return df


def perturb_inductance(df, level, seed, id_cols=None):
    """Add zero-mean Gaussian noise to the four raw coil inductances.

    Frank specified noise on the inductance values, so it is applied to L2 and L3
    here - before the differential is formed, not to the finished feature vector.

    Scale: sigma = level * |L_coil(f)|, the complex modulus of that coil at that
    frequency, shared by the real and imaginary channels of that coil but drawn
    independently for each. Using the modulus rather than each value's own magnitude
    matters because H_real passes through zero mid-band; a per-value scale would
    leave the zero crossing almost noise-free, exactly where ZCF is most fragile.

    Note the differential neither preserves nor uniformly amplifies this: at
    mu_r=200, rho=2e-7 the ratio |L_coil2|/|H| runs from 554x at the 750 Hz zero
    crossing down to below 1x on the low-frequency imaginary channels. One noise
    level on L is not one noise level in dL - 02_noise.ipynb reports the chain.

    Common random numbers: the standard normals are drawn once per `seed`, in a
    canonical row order, and only then scaled by `level`. Realisation r is therefore
    the same underlying draw at every noise level, which removes level-to-level
    sampling scatter from the trend and lets the levels be compared directly.

    level=0.0 returns the input values unchanged, bit-for-bit.

    Returns a new frame; the input is never modified.
    """
    if id_cols is None:
        id_cols = [c for c in df.columns if c not in COIL_COLS]

    out = df.copy()

    # Draw in a canonical order so a given seed means the same realisation
    # regardless of how the caller happened to sort the frame.
    order = np.lexsort([out[c].to_numpy() for c in reversed(id_cols)])

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(out), len(COIL_COLS)))

    amp = {c: np.hypot(out[f'{c}_real'].to_numpy(), out[f'{c}_imag'].to_numpy())
           for c in ('coil2', 'coil3')}

    for j, (col, coil) in enumerate(zip(COIL_COLS, COIL_OF)):
        noise = np.empty(len(out))
        noise[order] = z[:, j]              # canonical draw -> original row order
        out[col] = out[col].to_numpy() + level * amp[coil] * noise

    return out


def to_wide_features(df, index_cols, H):
    """Long form -> the 16-column H feature matrix, in the training column order.

    Mirrors src.data.reshape_wide but pivots on an arbitrary index (the liftoff
    sweep has no material columns), and ends with an explicit reindex to `H`.

    That last step is not cosmetic. reshape_wide inherits its column order from
    pd.pivot_table alphabetising values=['H_real','H_imag'], so the real order is
    the H_imag block first (375 -> 48000) and then the H_real block - not the order
    the source reads top to bottom. dense_models.ipynb measured what a silent
    transposition costs: 101-189 SD of displacement against 0.505 SD when correct,
    which would read as a physics result rather than a bug.
    """
    long = compute_differential(df.copy())
    wide = pd.pivot_table(long, index=index_cols, columns='frequency',
                          values=['H_real', 'H_imag'])
    wide = wide.reset_index()
    wide.columns = [f'{c[0]}_{c[1]}' if c[1] != '' else c[0] for c in wide.columns]

    missing = [c for c in H if c not in wide.columns]
    if missing:
        raise ValueError(f'feature columns absent after reshape: {missing}')

    out = wide[list(index_cols) + list(H)]
    assert list(out.columns[len(index_cols):]) == list(H), 'feature order does not match H'
    return out


# --- ZCF baseline -------------------------------------------------------------
# Moved verbatim from notebooks/zcf_baseline/01_zcf.ipynb (cells 6 and 13), with the
# calibration table passed in rather than closed over. Re-running that notebook must
# still reproduce 9.2% (mu|rho) and 7.6% (rho|mu); any drift means these were altered.
#
# The crossing is of H_real, the signed differential real part, as a function of
# frequency. src.data.compute_differential forms it as coil2 - coil3; the notebook
# prose says coil3 - coil2. The code is self-consistent and ZCF is invariant to a
# global sign flip, so the numbers are unaffected - but the prose is the wrong way round.

def zcf(group):
    g = group.sort_values('frequency')
    f = g['frequency'].to_numpy(); y = g['H_real'].to_numpy(); lf = np.log(f)
    idx = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    if len(idx) != 1:                    # 0 = no crossing, >1 = ambiguous
        return np.nan
    i = idx[0]
    lf0 = lf[i] + (0.0 - y[i]) * (lf[i+1] - lf[i]) / (y[i+1] - y[i])
    return float(np.exp(lf0))


def zcf_table(d, index_cols=('permeability', 'resistivity')):
    """One ZCF per group. NaNs are kept, not dropped.

    Under noise the 'exactly one in-band crossing' guard starts failing, and the
    failure rate is itself a result - dropping the NaNs here would compute MAPE on
    the surviving easy materials and flatter the baseline.
    """
    index_cols = list(index_cols)
    rows = [tuple(k if isinstance(k, tuple) else (k,)) + (zcf(g),)
            for k, g in d.groupby(index_cols)]
    return pd.DataFrame(rows, columns=index_cols + ['z'])


def predict_mu_given_rho(z_query, rho, calib):
    # np.isclose's default atol=1e-8 is only 10x below the 1e-7 resistivity grid spacing,
    # so this column select is correct but not comfortably so. Do not reuse the default
    # tolerance for anything comparing rho VALUES rather than picking a grid column --
    # at rho ~ 2e-7 the default atol alone makes 2.04e-7 compare equal to 2e-7.
    col = calib[np.isclose(calib['resistivity'], rho)].dropna(subset=['z']).sort_values('permeability')
    lz, lmu = np.log(col['z'].to_numpy()), np.log(col['permeability'].to_numpy())
    o = np.argsort(lz)                                  # ZCF ascending
    return float(np.exp(np.interp(np.log(z_query), lz[o], lmu[o])))


def predict_rho_given_mu(z_query, mu, calib, rhos):
    # ZCF(rho) at this permeability, built by interpolating each resistivity column
    # across log-permeability
    z_at_mu = []
    for r in rhos:
        c = calib[np.isclose(calib['resistivity'], r)].dropna(subset=['z']).sort_values('permeability')
        z_at_mu.append(np.exp(np.interp(np.log(mu), np.log(c['permeability'].to_numpy()),
                                                    np.log(c['z'].to_numpy()))))
    z_at_mu = np.array(z_at_mu)
    o = np.argsort(z_at_mu)                             # ZCF ascending
    return float(np.interp(z_query, z_at_mu[o], rhos[o]))   # LINEAR resistivity
