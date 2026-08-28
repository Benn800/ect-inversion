
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error


def extract_unique_freqs(df):
    return sorted({int(col.split('_')[-1]) for col in df.columns[2:]})

def round_sig(x, sig=3):
    return float(f"{x:.{sig}g}")


def compute_phase(df):
    new_d = {
    'permeability': df['permeability'],
    'resistivity': df['resistivity'],
    }
    frequencies = extract_unique_freqs(df)
    for f in frequencies:
        h_imag = df[f'H_imag_{f}']
        h_real = df[f'H_real_{f}']
        new_d[f'phase_{f}'] = np.mod(np.arctan2(h_imag, h_real), 2 * np.pi)

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


def plot_response_vs_permeability(df, component, figsize=(7, 5)):
    """Plot raw H_<component> vs permeability, averaged over resistivity,
    one line per frequency.
    """
    import matplotlib.pyplot as plt

    freqs = extract_unique_freqs(df)
    cols = [f'H_{component}_{f}' for f in freqs]
    g = df.groupby('permeability')[cols].mean()

    fig, ax = plt.subplots(figsize=figsize)
    for f, col in zip(freqs, cols):
        ax.plot(g.index, g[col] * 1e6, marker='o', label=f'{f} Hz')

    ax.set_xlabel('Permeability (μᵣ)')
    ax.set_ylabel(f'H_{component} (μH)')
    ax.set_title(f'H_{component} vs Permeability (averaged over resistivity)')
    ax.legend(fontsize=7, title='Frequency')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_response_step_vs_resistivity(df, component, permeability_values=(50, 500, 1000), figsize=None):
    """Plot step-to-step change in H_<component> vs resistivity, faceted by permeability
    (one subplot per value in `permeability_values`), one line per frequency. Each point is
    placed at the midpoint between the two resistivity values it was differenced from.
    """
    import matplotlib.pyplot as plt

    freqs = extract_unique_freqs(df)
    cols = [f'H_{component}_{f}' for f in freqs]

    fig, axes = plt.subplots(1, len(permeability_values),
                              figsize=figsize or (4 * len(permeability_values), 4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, mu in zip(axes, permeability_values):
        sub = df[df['permeability'] == mu].sort_values('resistivity').set_index('resistivity')
        step = sub[cols].diff() * 1e6
        midpoints = (sub.index.to_series().shift(1) + sub.index.to_series()) / 2
        for f, col in zip(freqs, cols):
            ax.plot(midpoints, step[col], marker='o', label=f'{f} Hz')
        ax.set_title(f'μᵣ = {mu}')
        ax.set_xlabel('Resistivity (ρ)')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f'Step size in H_{component} (μH)')
    axes[-1].legend(fontsize=7, title='Frequency')
    fig.suptitle(f'Step Size in H_{component} vs Resistivity (faceted by permeability)')
    fig.tight_layout()
    return fig, axes


def get_edge_groups(groups):
    unique_groups = np.unique(groups)
    return set(unique_groups[[0, -1]])


def run_logo_interpolation(model, X, y, groups, target_name):
    """LOGO CV over none-edge groups. Returns per-fold results + pooled metrics"""
    fold_results = []
    all_true_values = []
    all_predictions = []

    logo = LeaveOneGroupOut()
    edge_groups = get_edge_groups(groups)

    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = np.unique(groups[test_idx])[0]

        if held_out not in edge_groups:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            all_true_values.extend(y_test.values)
            all_predictions.extend(pred)

            result = ({
                'target': target_name,
                f'held_out_{target_name}': held_out,
                'true_values': [round_sig(v) for v in y_test],
                'predicted_values': [round_sig(v) for v in pred],
                'fold_mape': round((mean_absolute_percentage_error(y_test, pred))*100, 2),
                'fold_rmse': round_sig(root_mean_squared_error(y_test, pred))
            })
            if hasattr(model, 'feature_importances_'):
                result['feature_importances'] = dict(zip(X.columns, model.feature_importances_))

            fold_results.append(result)
    results_df = pd.DataFrame(fold_results)
    pooled_mape = round_sig(mean_absolute_percentage_error(all_true_values, all_predictions)*100)
    pooled_rmse = round_sig(root_mean_squared_error(all_true_values, all_predictions))

    return results_df, pooled_mape, pooled_rmse