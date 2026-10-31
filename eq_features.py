"""Feature code copied verbatim from the notebook (cell 0) so live data uses the exact training recipe."""
import numpy as np, pandas as pd
LOG10E = np.log10(np.e)

def gr_least_squares(mags):
    """Eqs. (3)-(5): least-squares fit of log10(N>=M) vs M."""
    order = np.sort(mags)[::-1]
    N = np.arange(1, len(order) + 1)
    logN = np.log10(N)
    M = order
    n = len(M)
    sM, slogN, sMlogN, sM2 = M.sum(), logN.sum(), (M * logN).sum(), (M ** 2).sum()
    denom = (sM ** 2 - n * sM2)
    if denom == 0:
        return np.nan, np.nan, logN, M
    b = (n * sMlogN - sM * slogN) / denom
    a = (slogN + b * sM) / n
    return a, b, logN, M

def gr_max_likelihood(mags):
    """Eqs. (6)-(7): Bender (1983) maximum-likelihood a,b."""
    mean_m, min_m = mags.mean(), mags.min()
    denom = mean_m - min_m
    if denom <= 0:
        return np.nan, np.nan
    b = LOG10E / denom
    n = len(mags)
    a = np.log10(n) + b * min_m
    return a, b

def window_features(win, t_now, full_df):
    M = win['magnitude'].values
    n = len(M)
    T_Days = max(win['t_days'].iloc[-1] - win['t_days'].iloc[0], 1e-6)          # Eq (1)

    a_lsq, b_lsq, logN, Msorted = gr_least_squares(M)                          # Eqs (3)-(5)
    a_lik, b_lik = gr_max_likelihood(M)                                       # Eqs (6)-(7)
    Mean_Mag = M.mean()                                                        # Eq (2)
    Energy = np.sum(np.sqrt(10 ** (11.8 + 1.5 * M))) / T_Days                  # Eq (8)

    # Eq (9): Habermann z-test, R1/R2 = event rate in 15d / 30d windows ending at t_now
    w15 = full_df[(full_df['t_days'] > t_now - 15) & (full_df['t_days'] <= t_now)]
    w30 = full_df[(full_df['t_days'] > t_now - 45) & (full_df['t_days'] <= t_now - 15)]
    n1, n2 = len(w15), len(w30)
    R1, R2 = n1 / 15.0, n2 / 30.0
    s1 = w15['magnitude'].std(ddof=0) if n1 > 1 else 0
    s2 = w30['magnitude'].std(ddof=0) if n2 > 1 else 0
    denom_z = np.sqrt((s1 ** 2 / max(n1, 1)) + (s2 ** 2 / max(n2, 1)))
    Z = (R1 - R2) / denom_z if denom_z > 0 else 0.0

    # Eq (10): max magnitude in last 7 days
    w7 = full_df[(full_df['t_days'] > t_now - 7) & (full_df['t_days'] <= t_now)]
    x6 = w7['magnitude'].max() if len(w7) else M[-1]

    # Eq (11): mean recurrence interval of "characteristic" events (M>=5.0 proxy)
    char = win[win['magnitude'] >= 5.0]
    if len(char) >= 2:
        dt = np.diff(char['t_days'].values)
        mu = dt.mean()
        c_val = dt.std(ddof=0) / mu if mu > 0 else 0.0                         # Eq (18)
    else:
        mu, c_val = T_Days, 0.0

    feats = {'T_Days': T_Days, 'b_lik': b_lik, 'b_lsq': b_lsq, 'a_lik': a_lik, 'a_lsq': a_lsq}

    # Eq (14): std dev of b-value
    feats['std_b_lik'] = 2.3 * b_lik ** 2 * np.sqrt(np.sum((M - Mean_Mag) ** 2) / (n * (n - 1))) if not np.isnan(b_lik) else np.nan
    feats['std_b_lsq'] = 2.3 * b_lsq ** 2 * np.sqrt(np.sum((M - Mean_Mag) ** 2) / (n * (n - 1))) if not np.isnan(b_lsq) else np.nan

    # Eq (17): T_Rec for M0 = 4.0 ... 6.0 in 0.1 steps (21 values) x 2 methods = 42 features
    for tag, a_, b_ in [('lik', a_lik, b_lik), ('lsq', a_lsq, b_lsq)]:
        for M0 in np.round(np.arange(4.0, 6.01, 0.1), 1):
            feats[f'T_Rec_{tag}_{M0}'] = T_Days / (10 ** (a_ - b_ * M0)) if not np.isnan(a_) else np.nan

    feats.update(Mean_Mag=Mean_Mag, Energy=Energy, Z_Value=Z, x6=x6, mu=mu, C=c_val)

    # Eq (12): probability of M>=6.5 (GR exceedance)
    feats['x7_mlk'] = 10 ** (a_lik - b_lik * 6.5) if not np.isnan(b_lik) else np.nan
    feats['x7_lsq'] = 10 ** (a_lsq - b_lsq * 6.5) if not np.isnan(b_lsq) else np.nan

    # Eq (13): deviation from GR law
    pred_lik = a_lik - b_lik * Msorted if not np.isnan(a_lik) else np.full(n, np.nan)
    pred_lsq = a_lsq - b_lsq * Msorted if not np.isnan(a_lsq) else np.full(n, np.nan)
    feats['NU_mlik'] = np.sum((logN - pred_lik) ** 2) / (n - 1)
    feats['NU_lsq'] = np.sum((logN - pred_lsq) ** 2) / (n - 1)

    # Eqs (15)-(16): magnitude deficit
    feats['M_def_Lik'] = (a_lik / b_lik) - M.max() if b_lik else np.nan
    feats['M_def_lsq'] = (a_lsq / b_lsq) - M.max() if b_lsq else np.nan

    feats['Actual_Mag'] = M[-1]
    return feats

def compute_all_features(df, n_window=50):
    rows, idxs = [], []
    for i in range(n_window - 1, len(df)):
        win = df.iloc[i - n_window + 1: i + 1]
        feats = window_features(win, df['t_days'].iloc[i], df)
        rows.append(feats); idxs.append(i)
    feat_df = pd.DataFrame(rows, index=idxs)
    feat_df['event_idx'] = idxs
    feat_df['t_days'] = df['t_days'].iloc[idxs].values
    return feat_df


KEY = ['year', 'month', 'day', 'hour', 'min', 'magnitude', 'lattitude', 'longitude']

def add_t_days(df):
    """Same t_days formula + de-duplication as load_catalog() in the notebook."""
    df = df.sort_values(['year', 'month', 'day', 'hour', 'min', 'second']).reset_index(drop=True)
    t = ((df['year'] - df['year'].min()) * 365.25 + (df['month'] - 1) * 30.44 + df['day']
         + df['hour'] / 24 + df['min'] / 1440 + df['second'] / 86400)
    df['t_days'] = t - t.min()
    return df.drop_duplicates(subset=KEY).reset_index(drop=True)

def live_features(df, last=1, n_window=50):
    """61-feature rows for the last `last` events (rolling window of 50 events, as in training)."""
    idx = list(range(len(df) - last, len(df)))
    rows = [window_features(df.iloc[i - n_window + 1:i + 1], df['t_days'].iloc[i], df) for i in idx]
    return pd.DataFrame(rows, index=idx)
