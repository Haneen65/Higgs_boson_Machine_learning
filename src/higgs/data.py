"""Data loading.

Two sources are supported, chosen automatically:

1. OFFICIAL  -> data/training.csv and data/test.csv from Kaggle / CERN Open Data
                (columns: EventId, 30 features, Weight, Label).
2. FALLBACK  -> data/_epfl_train.csv and data/_epfl_test.csv, a public copy of the
                same 250,000 training events (EventId 100000-349999) that contains
                Id, Prediction (s/b) and the 30 features, but NO Weight column.

With the fallback, per-event weights are *approximated* by one constant weight per
class, scaled so that the class totals match the totals of the official training
file.  See WEIGHT_MODE_NOTE and the report: AMS values computed this way are a proxy
and must not be compared with the Kaggle leaderboard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MISSING = -999.0
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Class totals of the weights in the official 250k training file (values recalled from
# the competition starting kit; NOT re-verified here because the official file could
# not be downloaded).  Only used by the fallback.
ASSUMED_SUM_SIGNAL_W = 691.9886
ASSUMED_SUM_BACKGROUND_W = 410999.8

FEATURES = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality", "DER_lep_eta_centrality", "PRI_tau_pt", "PRI_tau_eta",
    "PRI_tau_phi", "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi", "PRI_met",
    "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num", "PRI_jet_leading_pt",
    "PRI_jet_leading_eta", "PRI_jet_leading_phi", "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta", "PRI_jet_subleading_phi", "PRI_jet_all_pt",
]

WEIGHT_MODE_NOTE = ""


def load_data():
    """Return (train_df, test_df, info).

    train_df columns: EventId, <30 features>, Weight, Label (Label in {'s','b'}).
    test_df  columns: EventId, <30 features>.
    info: dict with 'official' (bool) and 'weight_note'.
    """
    global WEIGHT_MODE_NOTE
    off_tr, off_te = DATA / "training.csv", DATA / "test.csv"
    if off_tr.exists():
        tr = pd.read_csv(off_tr)
        te = pd.read_csv(off_te) if off_te.exists() else None
        if "KaggleSet" in tr.columns:           # CERN Open Data version of the file
            tr = tr[tr["KaggleSet"] == "t"].copy()
            if "KaggleWeight" in tr.columns:    # weights normalised for the Kaggle split
                tr["Weight"] = tr["KaggleWeight"]
            tr = tr.drop(columns=["KaggleSet", "KaggleWeight"], errors="ignore")
        note = "Official training.csv with true event weights."
        info = {"official": True, "weight_note": note}
    else:
        tr = pd.read_csv(DATA / "_epfl_train.csv").rename(columns={"Id": "EventId", "Prediction": "Label"})
        te = pd.read_csv(DATA / "_epfl_test.csv").rename(columns={"Id": "EventId"}).drop(columns=["Prediction"])
        is_s = tr["Label"].eq("s")
        w = np.where(is_s, ASSUMED_SUM_SIGNAL_W / is_s.sum(),
                     ASSUMED_SUM_BACKGROUND_W / (~is_s).sum())
        tr["Weight"] = w
        note = ("PROXY weights: the official Weight column was not available, so each class "
                "gets one constant weight (class totals "
                f"{ASSUMED_SUM_SIGNAL_W:.2f} signal / {ASSUMED_SUM_BACKGROUND_W:.2f} background).")
        info = {"official": False, "weight_note": note}
    WEIGHT_MODE_NOTE = note
    cols = ["EventId"] + FEATURES + ["Weight", "Label"]
    tr = tr[cols].reset_index(drop=True)
    if te is not None:
        te = te[["EventId"] + FEATURES].reset_index(drop=True)
    return tr, te, info
