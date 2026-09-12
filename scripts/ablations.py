"""
ablations.py
============
Defines the feature-group composition of every ablation in the study.

Each ablation ID maps to a list of feature-group names (see
FEATURE_FUNCS in sequence_features.py for the full set of valid group
names: length, gc, mono, di, kmer3, codon, uaug, kozak, cai, tai, mfe).

These lists are consumed by data_loader.load_and_prepare(), which slices
the cached full feature matrix down to only the requested groups —
no re-extraction of sequence features is needed per ablation.

Table reference (from thesis ablation design)
----------------------------------------------
A1   GC
A2   GC + Mono
A3   GC + Mono + Di
A4   GC + Mono + Di + 3-mer
A5   Length + GC + Mono + Di + 3-mer
A6   A5 + Codon
A7   A6 + uAUG
A8   A7 + Kozak
A9   A8 + CAI
A10  A8 + tAI
A11  A8 + CAI + tAI
A12  A11 + MFE
A13  Full model (== A12)
"""

_BASE = ["length", "gc", "mono", "di", "kmer3"]
_A8   = _BASE + ["codon", "uaug", "kozak"]

ABLATIONS = {
    "A1":  ["gc"],
    "A2":  ["gc", "mono"],
    "A3":  ["gc", "mono", "di"],
    "A4":  ["gc", "mono", "di", "kmer3"],
    "A5":  list(_BASE),
    "A6":  _BASE + ["codon"],
    "A7":  _BASE + ["codon", "uaug"],
    "A8":  list(_A8),
    "A9":  _A8 + ["cai"],
    "A10": _A8 + ["tai"],
    "A11": _A8 + ["cai", "tai"],
    "A12": _A8 + ["cai", "tai", "mfe"],
    "A13": _A8 + ["cai", "tai", "mfe"],  # full model, identical feature set to A12
}

# Sanity check: every group referenced here must be a real feature group.
_VALID_GROUPS = {"length", "gc", "mono", "di", "kmer3", "codon",
                  "uaug", "kozak", "cai", "tai", "mfe"}

for _aid, _groups in ABLATIONS.items():
    _unknown = set(_groups) - _VALID_GROUPS
    if _unknown:
        raise ValueError(f"ablations.py: {_aid} references unknown group(s) {_unknown}")
