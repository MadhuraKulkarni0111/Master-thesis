"""
ablations.py
============
Defines every ablation in the study as a "spec": which feature groups to
include, and optionally which regions to restrict those groups to.

Each ablation ID maps to a dict:
    {"groups": [list of group names], "regions": [list of regions] or None}

"groups" refers to FEATURE_FUNCS keys in sequence_features.py:
    length, gc, mono, di, kmer3, codon, uaug, kozak, cai, tai, mfe
"regions" refers to VALID_REGIONS in sequence_features.py:
    utr5, cds, utr3, full, window
regions=None means "no region restriction" (take the group's columns from
wherever they naturally live — this is what every A-series and CAI/tAI
ablation uses, since those groups already only matter in one place, or
are meant to span all regions).

data_loader.load_and_prepare() resolves a spec against the cached full
feature matrix using sequence_features.classify_column(), which tags
every column with its (group, region) purely from its name — no need to
re-run feature extraction per ablation.

===============================================================================
Family A — main progression (from thesis ablation design)
===============================================================================
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

CAI/tAI sub-study: for each base A1-A6, add CAI and/or tAI on top,
independent of the A7+ progression (tests codon-optimality/tRNA-adaptation
signal even before uAUG/Kozak are introduced).
    A{n}_cai, A{n}_tai, A{n}_cai_tai   for n in 1..6

===============================================================================
Family B — Location
===============================================================================
Restricts the compositional feature set (length + gc + mono + di + kmer3,
i.e. the same stack as A5) to one or more regions. Answers: how much TE
variation can be explained by sequence composition of THIS region alone?
Deliberately excludes mechanism-specific groups (codon/uaug/kozak/cai/tai)
— those are tested separately in Family C.

B1   5'UTR only
B2   CDS only
B3   3'UTR only
B4   5'UTR + CDS
B5   CDS + 3'UTR

===============================================================================
Family C — Mechanism
===============================================================================
C1   Initiation module   = uAUG + Kozak
C2   Elongation module   = Codon usage + CAI + tAI
C3   Initiation + elongation combined (C1 + C2)

===============================================================================
Family D — Representation
===============================================================================
D1   Generic 3-mers only
D2   Codons only
D3   3-mers + codons
D4   Start-centered (Kozak + MFE around the start codon) — compare against
     A4 (transcript-wide GC+mono+di+3-mer) as the other side of this
     representation contrast, rather than duplicating a "transcript-wide"
     entry under a new name.
===============================================================================
"""


def _spec(groups, regions=None):
    return {"groups": list(groups), "regions": list(regions) if regions else None}


# -----------------------------------------------------------------------------
# Family A
# -----------------------------------------------------------------------------
_BASE = ["length", "gc", "mono", "di", "kmer3"]
_A8   = _BASE + ["codon", "uaug", "kozak"]

ABLATIONS = {
    "A1":  _spec(["gc"]),
    "A2":  _spec(["gc", "mono"]),
    "A3":  _spec(["gc", "mono", "di"]),
    "A4":  _spec(["gc", "mono", "di", "kmer3"]),
    "A5":  _spec(_BASE),
    "A6":  _spec(_BASE + ["codon"]),
    "A7":  _spec(_BASE + ["codon", "uaug"]),
    "A8":  _spec(_A8),
    "A9":  _spec(_A8 + ["cai"]),
    "A10": _spec(_A8 + ["tai"]),
    "A11": _spec(_A8 + ["cai", "tai"]),
    "A12": _spec(_A8 + ["cai", "tai", "mfe"]),
    "A13": _spec(_A8 + ["cai", "tai", "mfe"]),  # full model, same set as A12
}

# CAI / tAI sub-study on top of A1-A6
for _base_id in ["A1", "A2", "A3", "A4", "A5", "A6"]:
    _base_groups = ABLATIONS[_base_id]["groups"]
    ABLATIONS[f"{_base_id}_cai"]     = _spec(_base_groups + ["cai"])
    ABLATIONS[f"{_base_id}_tai"]     = _spec(_base_groups + ["tai"])
    ABLATIONS[f"{_base_id}_cai_tai"] = _spec(_base_groups + ["cai", "tai"])

# -----------------------------------------------------------------------------
# Family B — Location
# -----------------------------------------------------------------------------
_B_GROUPS = ["length", "gc", "mono", "di", "kmer3"]

ABLATIONS["B1"] = _spec(_B_GROUPS, regions=["utr5"])
ABLATIONS["B2"] = _spec(_B_GROUPS, regions=["cds"])
ABLATIONS["B3"] = _spec(_B_GROUPS, regions=["utr3"])
ABLATIONS["B4"] = _spec(_B_GROUPS, regions=["utr5", "cds"])
ABLATIONS["B5"] = _spec(_B_GROUPS, regions=["cds", "utr3"])

# -----------------------------------------------------------------------------
# Family C — Mechanism
# -----------------------------------------------------------------------------
ABLATIONS["C1"] = _spec(["uaug", "kozak"])              # initiation module
ABLATIONS["C2"] = _spec(["codon", "cai", "tai"])         # elongation module
ABLATIONS["C3"] = _spec(["uaug", "kozak", "codon", "cai", "tai"])  # both

# -----------------------------------------------------------------------------
# Family D — Representation
# -----------------------------------------------------------------------------
ABLATIONS["D1"] = _spec(["kmer3"])
ABLATIONS["D2"] = _spec(["codon"])
ABLATIONS["D3"] = _spec(["kmer3", "codon"])
ABLATIONS["D4"] = _spec(["kozak", "mfe"])  # start-centered; compare vs A4

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
_VALID_GROUPS = {"length", "gc", "mono", "di", "kmer3", "codon",
                  "uaug", "kozak", "cai", "tai", "mfe"}
_VALID_REGIONS = {"utr5", "cds", "utr3", "full", "window"}

for _aid, _spec_dict in ABLATIONS.items():
    _unknown_groups = set(_spec_dict["groups"]) - _VALID_GROUPS
    if _unknown_groups:
        raise ValueError(f"ablations.py: {_aid} references unknown group(s) {_unknown_groups}")
    if _spec_dict["regions"] is not None:
        _unknown_regions = set(_spec_dict["regions"]) - _VALID_REGIONS
        if _unknown_regions:
            raise ValueError(f"ablations.py: {_aid} references unknown region(s) {_unknown_regions}")