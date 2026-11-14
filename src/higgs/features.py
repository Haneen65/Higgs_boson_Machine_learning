"""Missing-value strategies and physics-motivated feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, MISSING


def structural_missing_cols(df: pd.DataFrame) -> list[str]:
    """Features that contain at least one -999."""
    return [c for c in FEATURES if (df[c] == MISSING).any()]


def prepare(df: pd.DataFrame, strategy: str = "keep") -> pd.DataFrame:
    """Return a feature matrix (no EventId/Weight/Label) using a -999 strategy.

    keep       : leave -999 untouched (trees can split on it; fine for tree models)
    nan        : -999 -> NaN (HistGradientBoosting / LightGBM handle NaN natively)
    indicator  : -999 -> NaN plus one 0/1 column per affected feature
    jetgroup   : -999 -> NaN plus the categorical PRI_jet_num already present;
                 identical to 'nan' for trees, kept for readability in experiments
    """
    X = df[FEATURES].copy()
    miss_cols = structural_missing_cols(df)
    if strategy == "keep":
        return X
    ind = {f"{c}_undef": (X[c] == MISSING).astype(np.int8) for c in miss_cols}
    X = X.replace(MISSING, np.nan)
    if strategy in ("nan", "jetgroup"):
        return X
    if strategy == "indicator":
        return pd.concat([X, pd.DataFrame(ind, index=X.index)], axis=1)
    raise ValueError(strategy)


def add_physics_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineered features (expects NaN for undefined, i.e. output of prepare('nan'))."""
    X = X.copy()
    eps = 1e-6
    # visible mass vs MMC mass: MMC reconstructs the full di-tau mass including neutrinos
    X["ratio_mmc_vis"] = X["DER_mass_MMC"] / (X["DER_mass_vis"] + eps)
    # lepton/tau balance: signal taus and leptons come from one parent -> balanced pT
    X["tau_lep_pt_diff"] = X["PRI_tau_pt"] - X["PRI_lep_pt"]
    # missing energy relative to visible activity
    X["met_over_sumet"] = X["PRI_met"] / (X["PRI_met_sumet"] + eps)
    # total transverse momentum of the tau+lepton system relative to its scalar sum
    X["pt_h_over_sum_pt"] = X["DER_pt_h"] / (X["DER_sum_pt"] + eps)
    # angular separation tau-lepton in eta (the DER_deltar variable combines eta & phi)
    X["abs_deta_tau_lep"] = (X["PRI_tau_eta"] - X["PRI_lep_eta"]).abs()
    # transverse mass relative to MMC mass
    X["mt_over_mmc"] = X["DER_mass_transverse_met_lep"] / (X["DER_mass_MMC"] + eps)
    # leading-jet recoil against the tau+lepton system
    X["jet_lead_over_pt_h"] = X["PRI_jet_leading_pt"] / (X["DER_pt_h"] + eps)
    return X.replace([np.inf, -np.inf], np.nan)
