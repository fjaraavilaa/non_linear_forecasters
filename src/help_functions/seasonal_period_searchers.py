
"""Automated search for the seasonal period s.
 
Pipeline:
  1. Detrend + window the series, take the periodogram, pick peaks -> candidate periods.
  2. Score every candidate with STL seasonal strength (and ACF at lag s on the remainder-free signal).
  3. (Optional) re-rank the top candidates with your own out-of-sample score_fn.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, detrend
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
 
 
def candidate_periods(x, n_peaks=8, min_period=2, min_cycles=3):
    """Candidate integer periods from periodogram peaks."""
    x = detrend(np.asarray(x, float), type="linear")
    n = len(x)
    w = np.hanning(n)
    power = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(n)
 
    # keep periods in [min_period, n / min_cycles]
    ok = (freqs > 0) & (1 / np.where(freqs > 0, freqs, np.inf) >= min_period) \
         & (1 / np.where(freqs > 0, freqs, np.inf) <= n / min_cycles)
    idx = np.where(ok)[0]
    peaks, _ = find_peaks(power[idx])
    peaks = idx[peaks]
    top = peaks[np.argsort(power[peaks])[::-1][:n_peaks]]
 
    periods = np.unique(np.round(1 / freqs[top]).astype(int))
    return [int(p) for p in periods if p >= min_period]
 
 
def seasonal_strength(x, s):
    """STL-based seasonal strength in [0, 1] (Wang, Smith & Hyndman 2006)."""
    res = STL(np.asarray(x, float), period=s, robust=True).fit()
    var_r = np.var(res.resid)
    var_sr = np.var(res.seasonal + res.resid)
    return max(0.0, 1 - var_r / var_sr)
 
 
def acf_at_lag(x, s):
    """ACF at lag s after removing a slow trend, so long memory distorts it less."""
    x = np.asarray(x, float)
    trend = STL(x, period=s, robust=True).fit().trend
    return acf(x - trend, nlags=s, fft=True)[s]
 
 
def fast_scores(x, s):
    """O(n) seasonal strength and ACF at lag s, no STL.
    Detrend with a centred moving average of width s (removes trend and any cycle
    that divides s), estimate the seasonal profile by averaging over phases."""
    x = np.asarray(x, float)
    trend = pd.Series(x).rolling(s, center=True, min_periods=s // 2).mean().to_numpy()
    d = x - trend
    ok = ~np.isnan(d)
    d = d[ok]
    phase = (np.arange(len(x))[ok]) % s
    profile = np.bincount(phase, weights=d, minlength=s) / np.maximum(np.bincount(phase, minlength=s), 1)
    seas = profile[phase]
    resid = d - seas
    strength = max(0.0, 1 - np.var(resid) / np.var(seas + resid))
    a = acf(d, nlags=s, fft=True)[s]
    return strength, a


def find_seasonal_period(x, extra_candidates=(), n_peaks=8, top_k=3,
                         score_fn=None, verbose=True, max_period=None, fast=True,
                         min_strength=0.4, min_acf=0.2):
    """Return (best_s, table); best_s is None if no candidate is seasonal enough.
    A candidate qualifies if strength >= min_strength AND acf_s >= min_acf.
    score_fn(x, s) -> float (lower is better), e.g. CV error."""
    cands = sorted(set(candidate_periods(x, n_peaks)) | set(extra_candidates))
    if max_period is not None:
        cands = [s for s in cands if s <= max_period]
    rows = []
    for s in cands:
        if len(x) < 2 * s + 1:          # STL needs at least two full cycles
            continue
        if fast:
            st, a = fast_scores(x, s)
        else:
            st, a = seasonal_strength(x, s), acf_at_lag(x, s)
        rows.append({"s": s, "strength": st, "acf_s": a})
    table = pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)

    table["seasonal"] = (table["strength"] >= min_strength) & (table["acf_s"] >= min_acf)
    qualified = table[table["seasonal"]]

    if qualified.empty:
        best = None
    elif score_fn is not None:
        shortlist = qualified.head(top_k)["s"]
        scores = {int(s): score_fn(x, int(s)) for s in shortlist}
        table["cv_score"] = table["s"].map(scores)
        best = min(scores, key=scores.get)
    else:
        best = int(qualified.iloc[0]["s"])

    if verbose:
        print(table.to_string(index=False))
        if best is None:
            print(f"\nNo seasonal period found (need strength >= {min_strength} "
                  f"and acf_s >= {min_acf}). Consider a non-seasonal ARFIMA.")
        else:
            print(f"\nSelected s = {best}")
    return best, table
 
 
if __name__ == "__main__":
    # synthetic hourly data: daily (24) + weekly (168) cycles + long-memory-ish noise
    rng = np.random.default_rng(0)
    n = 24 * 7 * 20
    t = np.arange(n)
    x = (2.0 * np.sin(2 * np.pi * t / 24)
         + 1.0 * np.sin(2 * np.pi * t / 168)
         + 0.002 * t
         + np.cumsum(rng.normal(0, 0.05, n))
         + rng.normal(0, 0.5, n))
    find_seasonal_period(x)