"""
problem.py — the heteroscedastic Branin-like test problem (ground truth).

Pure definitions only: no run data needed. Mirrors the problem set up in
study_driver.m (x1 in [-5, 10], 5 categorical levels). Imported by results.py.
"""
import numpy as np

# ---- domain / problem constants ----
LB, UB = -5, 10
VAR_FCTR = np.array([15, 2, 8, 0, 10.])                       # actual value of each level
NOISE_MULS = np.array([1.00, 0.70, 0.90, 0.50, 1.20]) * 10    # per-level noise multiplier

# Canonical 12-config sweep (acf, param); folder tag matches run_sweep.acf_tag().
CONFIG_ORDER = [
    ("lcb", float("nan")), ("pi", float("nan")), ("ei", float("nan")),
    ("haei", 0.5), ("haei", 1.0), ("haei", 5.0),
    ("anpei", 0.2), ("anpei", 0.5), ("anpei", 0.8),
    ("rahbo", 0.5), ("rahbo", 1.0), ("rahbo", 5.0),
]
_KNOB = {"haei": "g", "rahbo": "a", "anpei": "b"}


# ---- noise-free objective and noise model ----
def f_true(x1, x2):
    """True noise-free objective (Branin-like). x2 is the level VALUE (not index)."""
    return ((x2 - 5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1 - 6)**2
            + 10*(1 - 1/(8*np.pi))*np.cos(x1) + 10)


def sigma(x1, level):
    """Heteroscedastic noise std at continuous x1 and 1-based categorical level."""
    return 0.135*np.exp((0.15*x1)**2) * NOISE_MULS[level-1]


# ---- ground-truth summaries ----
def ground_truth_min():
    """True noise-free global minimum over x1 in [LB, UB] and all levels."""
    x1 = np.linspace(LB, UB, 4000)
    return float(min(f_true(x1, v).min() for v in VAR_FCTR))


def true_opt_location():
    """(level, x1) of the true global optimum."""
    x1 = np.linspace(LB, UB, 4000); best = (np.inf, None, None)
    for lv, v in enumerate(VAR_FCTR, 1):
        fv = f_true(x1, v)
        if fv.min() < best[0]:
            best = (fv.min(), lv, x1[fv.argmin()])
    return best[1], best[2]                                   # level, x1


def true_min_per_category():
    """True noise-free minimum of f within each category level."""
    x1 = np.linspace(LB, UB, 4000)
    return np.array([f_true(x1, v).min() for v in VAR_FCTR])


# ---- config labels / tags / canonical keys ----
def label(acf, param):
    if acf == "haei":  return f"HAEI(γ={param:g})"
    if acf == "rahbo": return f"RAHBO(α={param:g})"
    if acf == "anpei": return f"ANPEI(β={param:g})"
    return acf.upper()


def acf_tag(acf, param):
    """Folder tag for an acquisition config (matches run_sweep.acf_tag())."""
    if param != param:                                        # NaN -> no knob (ei/lcb/pi)
        return acf
    return f"{acf}_{_KNOB.get(acf, 'p')}{param:g}"


def canon_cfg(acf, param):
    """Canonical config key: (acf, 'na' if NaN else rounded param). Matches the
    key used for loaded runs so ei/lcb/pi (param=NaN) compare correctly."""
    return (acf, "na" if param != param else round(float(param), 6))


# ---- journal-level styling: colorblind-safe, also readable in grayscale ----
# color + marker encode the acquisition FAMILY; linestyle encodes the PARAMETER.
# Colors are Okabe-Ito (colorblind-safe); yellow is dropped (poor on white).
FAMILY_COLOR = {
    "lcb": "#0072B2", "pi": "#E69F00", "ei": "#009E73",
    "haei": "#D55E00", "anpei": "#CC79A7", "rahbo": "#56B4E9",
}
FAMILY_MARKER = {
    "lcb": "v", "pi": "s", "ei": "o", "haei": "^", "anpei": "D", "rahbo": "P",
}
_LINESTYLES = ["-", "--", ":", "-."]          # low -> high parameter within a family
# sorted parameter values per family (for the linestyle rank)
_FAMILY_PARAMS = {}
for _a, _p in CONFIG_ORDER:
    if _p == _p:                               # skip NaN (the no-param baselines)
        _FAMILY_PARAMS.setdefault(_a, set()).add(round(float(_p), 6))
_FAMILY_PARAMS = {a: sorted(v) for a, v in _FAMILY_PARAMS.items()}


def style_for(acf, param):
    """(color, marker, linestyle) for a config: color+marker = family, linestyle =
    parameter rank within the family (no-param baselines use a solid line)."""
    color = FAMILY_COLOR.get(acf, "#444444")
    marker = FAMILY_MARKER.get(acf, "o")
    if param != param:                         # NaN -> singleton baseline
        ls = "-"
    else:
        params = _FAMILY_PARAMS.get(acf, [])
        p = round(float(param), 6)
        rank = params.index(p) if p in params else 0
        ls = _LINESTYLES[rank % len(_LINESTYLES)]
    return {"color": color, "marker": marker, "linestyle": ls}


# ---- selectable categorical palettes (one color per family, in FAMILY_ORDER) ----
# References (see the notebook): Okabe & Ito 2008 (CUD); Wong 2011 Nat. Methods;
# Tol 2021 (SRON colour schemes); Crameri et al. 2020 Nat. Commun.; ColorBrewer.
FAMILY_ORDER = list(FAMILY_COLOR)              # [lcb, pi, ei, haei, anpei, rahbo]

PALETTES = {
    "okabe_ito":   ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],  # CB-safe (default)
    "tol_bright":  ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"],  # CB-safe
    "tol_muted":   ["#332288", "#88CCEE", "#44AA99", "#117733", "#DDCC77", "#CC6677"],  # CB-safe
    "tol_vibrant": ["#0077BB", "#EE7733", "#009988", "#CC3311", "#EE3377", "#33BBEE"],  # CB-safe
    "dark2":       ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"],  # ColorBrewer
    "grayscale":   ["#000000", "#2F2F2F", "#555555", "#777777", "#999999", "#BBBBBB"],  # mono (use marks)
    "custom":      ["#A11AA1", "#1A80F4", "#00A7E5", "#00BB89", "#BF170B", "#E6A1E9"],  # user-supplied
}


def set_palette(palette):
    """Switch the per-family categorical colors. `palette` is a name in PALETTES or a
    list of >= len(FAMILY_ORDER) hex colors (assigned to families in FAMILY_ORDER)."""
    cols = PALETTES[palette] if isinstance(palette, str) else list(palette)
    if len(cols) < len(FAMILY_ORDER):
        raise ValueError(f"need >= {len(FAMILY_ORDER)} colors, got {len(cols)}")
    for fam, c in zip(FAMILY_ORDER, cols):
        FAMILY_COLOR[fam] = c
