"""
Perfiles energéticos de apilamiento nucleotídico y EDS.
Basado en parámetros nearest-neighbor SantaLucia (1998) y Turner (2004).
"""

import numpy as np
import pandas as pd

# ============================================================
# Parámetros nearest-neighbor (kcal/mol)
# ============================================================

STACKING_SANTALUCIA = {
    'AA': -1.00, 'AT': -0.88, 'AG': -1.28, 'AC': -1.44,
    'TA': -0.58, 'TT': -1.00, 'TG': -1.45, 'TC': -1.30,
    'GA': -1.30, 'GT': -1.44, 'GG': -1.84, 'GC': -2.17,
    'CA': -1.45, 'CT': -1.28, 'CG': -2.17, 'CC': -1.84,
}

STACKING_TURNER = {
    'AA': -0.93, 'AT': -1.10, 'AG': -2.08, 'AC': -2.24,
    'TA': -1.33, 'TT': -0.93, 'TG': -2.35, 'TC': -2.11,
    'GA': -2.35, 'GT': -2.24, 'GG': -3.26, 'GC': -3.42,
    'CA': -2.11, 'CT': -2.08, 'CG': -2.36, 'CC': -3.26,
}


def stacking_profile(sequence, params=None):
    """
    Calcula perfil de energía de apilamiento nucleótido a nucleótido.

    Args:
        sequence: Secuencia de DNA/RNA (str)
        params: Dict de parámetros dinucleótido→energía.
                Default: STACKING_SANTALUCIA

    Returns:
        np.array con energía por posición
    """
    if params is None:
        params = STACKING_SANTALUCIA

    seq = sequence.upper().replace('U', 'T')  # RNA → DNA si necesario
    n = len(seq)
    bonds = np.array([params.get(seq[i:i+2], 0.0) for i in range(n - 1)])
    profile = np.zeros(n)
    profile[0] = bonds[0]
    profile[-1] = bonds[-1]
    for i in range(1, n - 1):
        profile[i] = (bonds[i-1] + bonds[i]) / 2
    return profile


def compute_eds(wt_profile, mut_profile, mutation_nt_pos):
    """
    Calcula Energetic Disruption Score (41 sub-features).

    Args:
        wt_profile: Perfil energético wild-type (np.array)
        mut_profile: Perfil energético mutante (np.array)
        mutation_nt_pos: Posición del nucleótido mutado (0-indexed)

    Returns:
        dict con 41 features del EDS
    """
    diff = mut_profile - wt_profile
    abs_diff = np.abs(diff)
    n = len(diff)

    # Features globales
    f = {
        'eds_mean_abs_diff': np.mean(abs_diff),
        'eds_max_abs_diff': np.max(abs_diff),
        'eds_std_diff': np.std(diff),
        'eds_sum_abs_diff': np.sum(abs_diff),
        'eds_rms_diff': np.sqrt(np.mean(diff**2)),
        'eds_n_disrupted_01': int(np.sum(abs_diff > 0.1)),
        'eds_n_disrupted_02': int(np.sum(abs_diff > 0.2)),
        'eds_n_disrupted_05': int(np.sum(abs_diff > 0.5)),
        'eds_frac_disrupted': np.mean(abs_diff > 0.1),
    }

    # Features locales (ventanas)
    for window in [5, 10, 25, 50]:
        lo = max(0, mutation_nt_pos - window)
        hi = min(n, mutation_nt_pos + window)
        local = abs_diff[lo:hi]
        local_diff = diff[lo:hi]
        prefix = f'eds_local_{window}'
        f[f'{prefix}_mean'] = np.mean(local)
        f[f'{prefix}_max'] = np.max(local)
        f[f'{prefix}_std'] = np.std(local_diff)
        f[f'{prefix}_sum'] = np.sum(local)

    # Features del perfil mutante
    f['mut_mean_energy'] = np.mean(mut_profile)
    f['mut_std_energy'] = np.std(mut_profile)
    f['mut_min_energy'] = np.min(mut_profile)
    f['mut_max_energy'] = np.max(mut_profile)
    f['mut_range_energy'] = np.ptp(mut_profile)

    # Features del perfil WT
    f['wt_mean_energy'] = np.mean(wt_profile)
    f['wt_std_energy'] = np.std(wt_profile)

    # Features de forma
    f['eds_skewness'] = float(pd.Series(diff).skew())
    f['eds_kurtosis'] = float(pd.Series(diff).kurtosis())

    # Score compuesto
    f['eds_score'] = (
        f['eds_mean_abs_diff'] * 10 +
        f['eds_max_abs_diff'] * 5 +
        f['eds_local_10_mean'] * 8 +
        f['eds_n_disrupted_01'] * 0.01 +
        f['eds_rms_diff'] * 3
    )

    return f


def dual_profile(sequence):
    """
    Calcula perfiles con ambos modelos (SantaLucia + Turner).

    Returns:
        (profile_santalucia, profile_turner)
    """
    return (
        stacking_profile(sequence, STACKING_SANTALUCIA),
        stacking_profile(sequence, STACKING_TURNER),
    )


def repeat_expansion_profile(repeat_unit, n_repeats, flanking_5='', flanking_3='',
                              params=None):
    """
    Calcula perfil energético para una expansión de repeticiones.
    Útil para DM1 (CTG), Fragile X (CGG), Huntington (CAG), etc.

    Args:
        repeat_unit: Unidad de repetición (e.g. 'CTG')
        n_repeats: Número de repeticiones
        flanking_5: Secuencia flanqueante 5'
        flanking_3: Secuencia flanqueante 3'
        params: Parámetros nearest-neighbor

    Returns:
        np.array con perfil energético completo
    """
    sequence = flanking_5 + (repeat_unit * n_repeats) + flanking_3
    return stacking_profile(sequence, params)
