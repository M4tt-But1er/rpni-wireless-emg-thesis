"""Feature families of docs/family_ablation_declared_protocol.md: 10 leaf families in 5 branches covering every column of a wide table (set C).
Used by family_ablation.py, family_backward_elimination.py and the bitrate grid (wide set W = the leaves kept by the elimination)."""
import collections
import re

from wide_features import EXTRA_NAMES, FEATURE_NAMES, C_ONLY_NAMES, BP_NAMES, WP_NAMES, DYN_NAMES

# leaf -> (branch, per-channel type names). Wavelet types of set A (wav_energy_A4 ...) are matched by regex because the level depends on the window length.
_AMPL = ["MAV", "RMS", "WL", "IEMG", "Var", "MaxAbs", "DASDV", "LogDet"]
_SHAPE = ["ZC", "SSC", "WAMP", "HjMob", "HjCx", "Skew", "Kurtosis", "ShannonEntropy"]
_SPEC = ["MNF", "MDF", "PeakFreq", "BandPowerRatio", "SpecSpread", "SpecFlat", "SpecSkew", "SpecKurt"]
_AR = ["AR1", "AR2", "AR3", "AR4", "CC1", "CC2", "CC3", "CC4"]
LEAVES = collections.OrderedDict([
    ("amplitude", ("time_domain", _AMPL)),
    ("shape_complexity", ("time_domain", _SHAPE)),
    ("spectral_summary", ("frequency_domain", _SPEC)),
    ("subband_power", ("frequency_domain", list(BP_NAMES))),
    ("wavelet", ("frequency_domain", list(WP_NAMES))),   # plus the regex for set A's wav_* columns
    ("ar_cepstral", ("model_based", _AR)),
    ("window_change", ("temporal_context", list(DYN_NAMES))),
    ("pair_xcorr", ("cross_channel", [])),
    ("rms_share", ("cross_channel", [])),
    ("pair_bandcorr", ("cross_channel", [])),
])
BRANCHES = collections.OrderedDict()
for _leaf, (_br, _) in LEAVES.items():
    BRANCHES.setdefault(_br, []).append(_leaf)
MULTI_LEAF_BRANCHES = [b for b, leaves in BRANCHES.items() if len(leaves) > 1]

_TYPE_TO_LEAF = {t: leaf for leaf, (_, types) in LEAVES.items() for t in types}
_TYPES_LONGEST_FIRST = sorted(_TYPE_TO_LEAF, key=len, reverse=True)
_WAV_RE = re.compile(r"_wav_(energy|entropy)_[AD]\d$")


def leaf_of(col):
    """Leaf family of one feature column name."""
    if col.startswith("xcorr_"):
        return "pair_xcorr"
    if col.startswith("rmsshare_"):
        return "rms_share"
    if col.startswith("bandcorr"):
        return "pair_bandcorr"
    if _WAV_RE.search(col):
        return "wavelet"
    for t in _TYPES_LONGEST_FIRST:
        if col.endswith("_" + t):
            return _TYPE_TO_LEAF[t]
    raise ValueError(f"column without a family: {col}")


def branch_of(col):
    return LEAVES[leaf_of(col)][0]


def check_complete():
    """Every type known to wide_features belongs to exactly one leaf (called by the tests)."""
    known = set(FEATURE_NAMES) | set(EXTRA_NAMES) | set(C_ONLY_NAMES)
    in_leaves = {t for _, (_, ts) in LEAVES.items() for t in ts} | {n for n in FEATURE_NAMES if n.startswith("wav_")}
    missing = known - in_leaves
    assert not missing, missing
    return True


def subsets_of(cols, a_cols):
    """The 28 feature subsets of the protocol as an ordered dict name -> list of columns. `a_cols` = the original-type columns (set A) of the same table."""
    leaf = {c: leaf_of(c) for c in cols}
    branch = {c: LEAVES[leaf[c]][0] for c in cols}
    aset = set(a_cols)
    out = collections.OrderedDict()
    out["A"] = [c for c in cols if c in aset]
    out["C"] = list(cols)
    for l in LEAVES:
        out[f"drop:{l}"] = [c for c in cols if leaf[c] != l]
    for l in LEAVES:
        out[f"add:{l}"] = [c for c in cols if c in aset or leaf[c] == l]
    for b in MULTI_LEAF_BRANCHES:
        out[f"drop:{b}"] = [c for c in cols if branch[c] != b]
    for b in MULTI_LEAF_BRANCHES:
        out[f"add:{b}"] = [c for c in cols if c in aset or branch[c] == b]
    return out
