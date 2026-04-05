"""
Does Knowing One's Aggressor Reduce Reporting to Authorities?
A Causal Inference Analysis of the National Crime Victimization Survey (1993–2023)

Author  : Clément HENNET
Date    : 2025
Dataset : Bureau of Justice Statistics – NCVS Incident-Level Data
API     : https://data.ojp.usdoj.gov/resource/gcuy-rt5g.json

Research Question
-----------------
Does having a prior relationship with one's aggressor causally reduce the
probability of reporting the crime to the police?

Strategy
--------
1. Descriptive statistics and weighted visualizations (survey-weighted).
2. Naive logistic regression (OLS-analogue for binary outcomes) to assess
   conditional associations — interpreted cautiously due to selection bias.
3. Propensity Score Matching (PSM) to estimate the Average Treatment Effect
   on the Treated (ATT): knowing one's aggressor (treatment = 1) vs. not
   knowing (control = 0), matched within crime type strata to remove the
   most obvious confound.
4. Post-matching balance diagnostics (Standardized Mean Differences).
5. Outcome regression on the matched sample for ATT estimation with
   covariate adjustment.

Identification Assumptions
---------------------------
- Conditional Independence (CIA / Unconfoundedness): conditional on observed
  covariates (sex, marital status, minority status, income), treatment
  assignment (knowing the aggressor) is independent of potential outcomes.
  This is a strong assumption — unmeasured confounders (e.g., relationship
  quality, prior victimization) may still bias estimates.
- Common Support: treated and control units overlap sufficiently in
  propensity score distributions — verified graphically.
- SUTVA: each unit's outcome is unaffected by other units' treatment status.

Notes on Sampling Weights
--------------------------
Survey weights (wgtviccy) are used only for descriptive/population-level
figures. PSM targets the ATT on the observed sample, not the population ATE,
so unweighted propensity score estimation is appropriate (Austin et al. 2016;
Lenis et al. 2019). See README for full discussion.
"""

# =============================================================================
# 0. IMPORTS & PACKAGES
# =============================================================================

import os
import warnings
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.iolib.summary2 import summary_col

warnings.filterwarnings("ignore")

# Output directory for saved figures
os.makedirs("outputs/figures", exist_ok=True)

# Plot aesthetics
PALETTE = {"treat": "#1a3a5c", "control": "#c0392b", "neutral": "#7f8c8d"}
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# =============================================================================
# 1. IMPORT DATA
# =============================================================================

API_URL  = "https://api.ojp.gov/bjsdataset/v1/gcuy-rt5g.json"
YEARS    = list(range(1993, 2024))
CACHE    = "victimization_data.csv"

def fetch_ncvs(url: str, years: list[int], cache_path: str) -> pd.DataFrame:
    """
    Download NCVS Select Personal Victimization records from the BJS API
    for the requested years.

    Parameters
    ----------
    url        : BJS REST API endpoint (api.ojp.gov).
    years      : List of calendar years to keep.
    cache_path : Local CSV path for caching.

    Returns
    -------
    pd.DataFrame with raw API records filtered to the requested years.
    """
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached data from '{cache_path}'.")
        return pd.read_csv(cache_path, dtype=str)

    try:
        response = requests.get(url, params={"$limit": 100_000}, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise SystemExit(
            "\n[ERROR] Could not reach the BJS API. Possible causes:\n"
            "  1. No internet connection.\n"
            "  2. The domain 'api.ojp.gov' is blocked on your network.\n"
            "  3. The API endpoint has changed again — check:\n"
            "     https://bjs.ojp.gov/national-crime-victimization-survey-ncvs-api\n\n"
            "Manual fallback: download the dataset directly from\n"
            "  https://bjs.ojp.gov/data-collection/ncvs\n"
            "and save it as 'victimization_data.csv' in this directory.\n"
            f"Original error: {e}"
        )

    data = response.json()
    if not data:
        raise ValueError("API returned an empty response.")

    df = pd.DataFrame(data)

    # Filter to requested years (year column is a string in the API response)
    year_strs = {str(y) for y in years}
    if "year" in df.columns:
        df = df[df["year"].isin(year_strs)]

    df.to_csv(cache_path, index=False)
    print(f"[INFO] Fetched {len(df):,} records ({min(years)}–{max(years)}) "
          f"→ saved to '{cache_path}'.")
    return df


raw = fetch_ncvs(API_URL, YEARS, CACHE)

# =============================================================================
# 2. DATA CLEANING & VARIABLE CONSTRUCTION
# =============================================================================

# --- Numeric conversions -------------------------------------------------------
raw["wgtviccy"] = pd.to_numeric(raw["wgtviccy"], errors="coerce")   # survey weight
raw["newoff"]   = pd.to_numeric(raw["newoff"],   errors="coerce")   # crime type code

# --- Binary / dummy variables -------------------------------------------------

def _relationship(v: str) -> float:
    """
    Treatment indicator.
    1 = victim knew aggressor (intimate, relative, or acquaintance)
    0 = stranger
    NaN = relationship unknown / non-applicable
    """
    if v in ("1", "2", "3"):
        return 1.0
    if v == "4":
        return 0.0
    return np.nan

def _married(v: str) -> float:
    """1 = currently married; 0 = all other statuses."""
    if v == "2":
        return 1.0
    if v in ("1", "3", "4", "5"):
        return 0.0
    return np.nan

def _female(v: str) -> float:
    """1 = female; 0 = male."""
    if v == "2":
        return 1.0
    if v == "1":
        return 0.0
    return np.nan

def _minor(v) -> float:
    """
    1 = respondent is 12–17 (NCVS age recode = 1)
    0 = 18 or older (age recodes 2–6)
    """
    try:
        v = int(v)
    except (ValueError, TypeError):
        return np.nan
    if v == 1:
        return 1.0
    if v in (2, 3, 4, 5, 6):
        return 0.0
    return np.nan

def _reported(v: str) -> float:
    """1 = notified police; 0 = did not notify."""
    if v == "1":
        return 1.0
    if v == "2":
        return 0.0
    return np.nan

def _low_income(v: str) -> float:
    """
    1 = household income < $35,000 (categories 1–4)
    0 = household income ≥ $35,000 (categories 5–7)
    """
    if v in ("1", "2", "3", "4"):
        return 1.0
    if v in ("5", "6", "7"):
        return 0.0
    return np.nan


raw["relationship_dummy"]  = raw["direl"].apply(_relationship)
raw["married_dummy"]       = raw["marital"].apply(_married)
raw["female_dummy"]        = raw["sex"].apply(_female)
raw["minor_dummy"]         = raw["ager"].apply(_minor)
raw["reported_dummy"]      = raw["notify"].apply(_reported)
raw["lower_income_dummy"]  = raw["hincome1"].apply(_low_income)

# --- Select working columns & drop NaNs ---------------------------------------

COLS = [
    "relationship_dummy",
    "married_dummy",
    "female_dummy",
    "minor_dummy",
    "reported_dummy",
    "lower_income_dummy",
    "wgtviccy",
    "newoff",
    "year",
]

df = raw[COLS].copy()

# Report missingness before dropping
miss = df.isna().sum()
print("\n[Missing values per column]")
print(miss.to_string())
print(f"\nDropping {df.shape[0] - df.dropna().shape[0]:,} rows with any NaN.")

df = df.dropna().copy()
df["newoff"] = df["newoff"].astype(int)
print(f"[INFO] Working dataset: {len(df):,} observations.\n")

# =============================================================================
# 3. LABEL DICTIONARIES
# =============================================================================

CRIME_LABELS = {
    1: "Rape / Sexual Assault",
    2: "Robbery",
    3: "Aggravated Assault",
    4: "Simple Assault",
    5: "Personal Theft / Larceny",
}

CRIME_LABELS_SHORT = {
    1: "Rape/Sexual",
    2: "Robbery",
    3: "Aggravated",
    4: "Simple Assault",
    5: "Pers. Theft",
}

RELATIONSHIP_LABELS = {1: "Knew Aggressor", 0: "Did Not Know"}
REPORTED_LABELS     = {1: "Reported", 0: "Not Reported"}

COVARIATES = ["married_dummy", "female_dummy", "minor_dummy", "lower_income_dummy"]
COVARIATE_LABELS = {
    "married_dummy":      "Married",
    "female_dummy":       "Female",
    "minor_dummy":        "Minor (12–17)",
    "lower_income_dummy": "Low Income (<$35k)",
}

# =============================================================================
# 4. DESCRIPTIVE VISUALIZATIONS (survey-weighted)
# =============================================================================

def savefig(name: str) -> None:
    path = f"outputs/figures/{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"[SAVED] {path}")


# --- Figure 1: Weighted composition of crime types ---------------------------

wgt_by_crime = (
    df.groupby("newoff")["wgtviccy"].sum()
    / df["wgtviccy"].sum()
).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
colors = sns.color_palette("Blues_d", len(wgt_by_crime))
bars   = ax.barh(
    [CRIME_LABELS[i] for i in wgt_by_crime.index],
    wgt_by_crime.values * 100,
    color=colors, edgecolor="white",
)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_xlabel("Share of Total Victimizations (%)")
ax.set_title("Weighted Composition of Victimization Types\nNCVS 1993–2023", fontweight="bold")
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
sns.despine(left=True)
ax.set_xlim(0, wgt_by_crime.max() * 100 * 1.18)
plt.tight_layout()
savefig("fig1_crime_composition")


# --- Figure 2: Evolution of victimization counts over time (log scale) -------

crime_year = (
    df.groupby(["newoff", "year"])["wgtviccy"].sum()
    .unstack(fill_value=0)
)
# Sort by total volume, descending
crime_year = crime_year.loc[crime_year.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 5))
palette = sns.color_palette("tab10", len(crime_year))

for idx, (crime_id, row) in enumerate(crime_year.iterrows()):
    ax.plot(
        row.index.astype(str),
        row.values,
        marker="o", markersize=4,
        label=CRIME_LABELS[int(crime_id)],
        color=palette[idx],
    )

ax.set_yscale("log")
ax.set_title("Victimization Trends by Crime Type\nNCVS 1993–2023 (survey-weighted, log scale)", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Weighted Count (log scale)")
ax.set_xticks(ax.get_xticks()[::3])
plt.xticks(rotation=45)
ax.legend(title="Crime Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
savefig("fig2_crime_trends")


# --- Figure 3: Victim–offender relationship by crime type (stacked bar) ------

rel_crime = (
    df.groupby(["newoff", "relationship_dummy"])["wgtviccy"].sum()
    .unstack(fill_value=0)
)
rel_crime_pct = rel_crime.div(rel_crime.sum(axis=1), axis=0) * 100
rel_crime_pct = rel_crime_pct.loc[
    rel_crime_pct.sum(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(9, 5))
rel_crime_pct.plot(
    kind="barh", stacked=True,
    color=[PALETTE["control"], PALETTE["treat"]],
    ax=ax, edgecolor="white", width=0.6,
)
ax.set_xlabel("Share (%)")
ax.set_title("Victim–Offender Relationship by Crime Type\nNCVS 1993–2023 (survey-weighted)", fontweight="bold")
ax.set_yticklabels([CRIME_LABELS[int(i)] for i in rel_crime_pct.index])
ax.legend(
    labels=[RELATIONSHIP_LABELS[int(k)] for k in rel_crime_pct.columns],
    title="Relationship", bbox_to_anchor=(1.01, 1), loc="upper left",
)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
sns.despine(left=True)
plt.tight_layout()
savefig("fig3_relationship_by_crime")


# --- Figure 4: Reporting rate by crime type -----------------------------------

rep_crime = (
    df.groupby(["newoff", "reported_dummy"])["wgtviccy"].sum()
    .unstack(fill_value=0)
)
rep_crime_pct = rep_crime.div(rep_crime.sum(axis=1), axis=0) * 100
rep_crime_pct = rep_crime_pct.loc[
    rep_crime_pct.sum(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(9, 5))
rep_crime_pct.plot(
    kind="barh", stacked=True,
    color=["#e74c3c", "#2ecc71"],
    ax=ax, edgecolor="white", width=0.6,
)
ax.set_xlabel("Share (%)")
ax.set_title("Police Reporting Rate by Crime Type\nNCVS 1993–2023 (survey-weighted)", fontweight="bold")
ax.set_yticklabels([CRIME_LABELS[int(i)] for i in rep_crime_pct.index])
ax.legend(
    labels=[REPORTED_LABELS[int(k)] for k in rep_crime_pct.columns],
    title="Reported to Police", bbox_to_anchor=(1.01, 1), loc="upper left",
)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
sns.despine(left=True)
plt.tight_layout()
savefig("fig4_reporting_rate")


# --- Figure 5: Relationship with aggressor among non-reporters ---------------

nonrep_rel = (
    df.query("reported_dummy == 0")
    .groupby(["newoff", "relationship_dummy"])["wgtviccy"].sum()
    .unstack(fill_value=0)
)
nonrep_rel_pct = nonrep_rel.div(nonrep_rel.sum(axis=1), axis=0) * 100
nonrep_rel_pct = nonrep_rel_pct.loc[
    nonrep_rel_pct.sum(axis=1).sort_values(ascending=False).index
]

fig, ax = plt.subplots(figsize=(9, 5))
nonrep_rel_pct.plot(
    kind="barh", stacked=True,
    color=[PALETTE["control"], PALETTE["treat"]],
    ax=ax, edgecolor="white", width=0.6,
)
ax.set_xlabel("Share (%)")
ax.set_title("Victim–Offender Relationship Among Unreported Crimes\nNCVS 1993–2023 (survey-weighted)", fontweight="bold")
ax.set_yticklabels([CRIME_LABELS[int(i)] for i in nonrep_rel_pct.index])
ax.legend(
    labels=[RELATIONSHIP_LABELS[int(k)] for k in nonrep_rel_pct.columns],
    title="Relationship", bbox_to_anchor=(1.01, 1), loc="upper left",
)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
sns.despine(left=True)
plt.tight_layout()
savefig("fig5_unreported_relationship")


# =============================================================================
# 5. PRE-MATCHING BALANCE CHECK
# =============================================================================

"""
Before estimating the propensity score, we verify that the treated (knew aggressor)
and control (did not know) groups differ meaningfully on observed covariates.
We use:
  (a) a group-means table for inspection, and
  (b) two-sided z-tests for proportions (all covariates are binary)

The z-test is appropriate here since all variables are binary proportions.
"""

print("=" * 60)
print("PRE-MATCHING COVARIATE BALANCE")
print("=" * 60)

# --- 5a. Group means table ---------------------------------------------------

psm_df = df[COVARIATES + ["relationship_dummy", "reported_dummy", "newoff"]].copy()

balance_pre = (
    psm_df.groupby("relationship_dummy")[COVARIATES + ["reported_dummy"]]
    .mean()
    .T.rename(columns={0: "Did Not Know (Control)", 1: "Knew Aggressor (Treated)"})
)
balance_pre["Difference"] = (
    balance_pre["Knew Aggressor (Treated)"] - balance_pre["Did Not Know (Control)"]
)
balance_pre.index = [COVARIATE_LABELS.get(i, i) for i in balance_pre.index[:-1]] + ["Reported to Police"]

print("\nCovariate means by treatment group (unweighted):")
print(balance_pre.round(3).to_string())

# --- 5b. Z-tests for proportions ---------------------------------------------

treated_pre = psm_df[psm_df["relationship_dummy"] == 1]
control_pre  = psm_df[psm_df["relationship_dummy"] == 0]

print("\nTwo-sided z-tests for equality of proportions (H0: p_treated = p_control):")
print(f"{'Variable':<25} {'Z-stat':>8} {'p-value':>10} {'Sig.':>5}")
print("-" * 55)

for var in COVARIATES:
    z, p = proportions_ztest(
        [treated_pre[var].sum(), control_pre[var].sum()],
        [treated_pre[var].count(), control_pre[var].count()],
    )
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    label = COVARIATE_LABELS.get(var, var)
    print(f"{label:<25} {z:>8.2f} {p:>10.4f} {sig:>5}")


# --- 5c. Visual: group means bar chart ---------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))
balance_plot = balance_pre.drop("Reported to Police").drop(columns="Difference")
x = np.arange(len(balance_plot))
w = 0.35
ax.bar(x - w/2, balance_plot["Did Not Know (Control)"],   w, label="Did Not Know", color=PALETTE["control"], alpha=0.85)
ax.bar(x + w/2, balance_plot["Knew Aggressor (Treated)"], w, label="Knew Aggressor", color=PALETTE["treat"],   alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(balance_plot.index, rotation=20, ha="right")
ax.set_ylabel("Mean")
ax.set_title("Covariate Means by Treatment Group (Pre-Matching)", fontweight="bold")
ax.legend()
sns.despine()
plt.tight_layout()
savefig("fig6_prebalance_means")


# =============================================================================
# 6. NAIVE LOGISTIC REGRESSION (pre-matching, full sample)
# =============================================================================

"""
Naive logistic regression on the full (unmatched) sample.

Interpretation
--------------
Coefficients are log-odds. A positive coefficient on 'relationship_dummy'
would imply that knowing one's aggressor is associated with higher reporting
odds — but this could be entirely driven by compositional differences across
crime types and victim demographics. These estimates do NOT have a causal
interpretation.
"""

print("\n" + "=" * 60)
print("NAIVE LOGISTIC REGRESSION (unmatched sample)")
print("=" * 60)

REGRESSORS = ["relationship_dummy"] + COVARIATES


def fit_logit(data: pd.DataFrame, y_col: str, x_cols: list):
    y = data[y_col]
    X = sm.add_constant(data[x_cols])
    return sm.Logit(y, X).fit(disp=False)


logit_full = fit_logit(df, "reported_dummy", REGRESSORS)

# Per-crime-type regressions
naive_by_crime = {}
for ct in sorted(df["newoff"].unique()):
    naive_by_crime[ct] = fit_logit(df[df["newoff"] == ct], "reported_dummy", REGRESSORS)

model_names_naive = ["Pooled"] + [CRIME_LABELS_SHORT[ct] for ct in sorted(naive_by_crime)]

naive_table = summary_col(
    [logit_full] + [naive_by_crime[ct] for ct in sorted(naive_by_crime)],
    stars=True,
    model_names=model_names_naive,
    float_format="%0.3f",
    info_dict={"N": lambda x: f"{int(x.nobs):,}"},
)
print(naive_table)
print(
    "\nNote: These are log-odds coefficients (not marginal effects). "
    "Causal interpretation requires the matching procedure below."
)


# =============================================================================
# 7. PROPENSITY SCORE MATCHING
# =============================================================================

"""
Strategy
--------
We estimate the propensity score — Pr(knew aggressor | X) — using logistic
regression on the four observed covariates.  Matching is done within crime-
type strata (newoff) via 1:1 nearest-neighbor matching WITHOUT replacement
(greedy). Stratified matching removes crime-type as a confound, since the
distribution of victim–offender relationships varies substantially across
crime categories.

ATT vs. ATE
-----------
We target the ATT: the effect of knowing one's aggressor on reporting,
for victims who knew their aggressor.  We want to understand the barrier
to reporting faced by victims with a prior relationship to their attacker.
"""

print("\n" + "=" * 60)
print("PROPENSITY SCORE ESTIMATION")
print("=" * 60)

psm_work = df[COVARIATES + ["relationship_dummy", "reported_dummy", "newoff"]].copy()

X_ps = psm_work[COVARIATES]
y_ps = psm_work["relationship_dummy"]

log_reg = LogisticRegression(max_iter=1000, solver="lbfgs")
log_reg.fit(X_ps, y_ps)

psm_work["propensity_score"] = log_reg.predict_proba(X_ps)[:, 1]

print(f"\nPropensity score summary:")
print(psm_work.groupby("relationship_dummy")["propensity_score"].describe().round(3).to_string())


# --- Figure 7: Common support ------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))
for val, label, color in [
    (1, "Knew Aggressor (Treated)",  PALETTE["treat"]),
    (0, "Did Not Know (Control)",    PALETTE["control"]),
]:
    subset = psm_work[psm_work["relationship_dummy"] == val]["propensity_score"]
    ax.hist(subset, bins=40, alpha=0.55, label=label, color=color, edgecolor="none", density=True)
    subset.plot.kde(ax=ax, color=color, linewidth=2)

ax.set_xlabel("Estimated Propensity Score")
ax.set_ylabel("Density")
ax.set_title("Common Support: Propensity Score Distributions\n(Pre-Matching)", fontweight="bold")
ax.legend()
sns.despine()
plt.tight_layout()
savefig("fig7_common_support")


# --- 7a. 1:1 nearest-neighbour matching within crime-type strata -------------

treated_all = psm_work[psm_work["relationship_dummy"] == 1]
control_all  = psm_work[psm_work["relationship_dummy"] == 0]

matched_controls = []

for ct in psm_work["newoff"].unique():
    t_ct = treated_all[treated_all["newoff"] == ct]
    c_ct = control_all[control_all["newoff"] == ct]
    if t_ct.empty or c_ct.empty:
        continue
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    nn.fit(c_ct[["propensity_score"]])
    distances, indices = nn.kneighbors(t_ct[["propensity_score"]])
    matched_controls.append(c_ct.iloc[indices.flatten()])

matched_data = pd.concat([treated_all, pd.concat(matched_controls)])

print(f"\n[INFO] Matched sample: {len(matched_data):,} obs "
      f"({len(treated_all):,} treated + {len(pd.concat(matched_controls)):,} matched controls).")


# =============================================================================
# 8. POST-MATCHING BALANCE DIAGNOSTICS
# =============================================================================

"""
We plot Standardized Mean Differences (SMDs) before and after matching.
"""

def smd(data: pd.DataFrame, vars_: list) -> dict:
    """Compute pooled-SD standardized mean difference for binary variables."""
    t = data[data["relationship_dummy"] == 1]
    c = data[data["relationship_dummy"] == 0]
    result = {}
    for v in vars_:
        mu_t, mu_c = t[v].mean(), c[v].mean()
        sd_pooled  = np.sqrt((t[v].var() + c[v].var()) / 2)
        result[v]  = abs(mu_t - mu_c) / sd_pooled if sd_pooled > 0 else 0.0
    return result


smd_pre  = smd(psm_work,   COVARIATES)
smd_post = smd(matched_data, COVARIATES)

print("\n" + "=" * 60)
print("POST-MATCHING BALANCE (Standardized Mean Differences)")
print("=" * 60)
print(f"{'Covariate':<25} {'SMD Pre':>10} {'SMD Post':>10} {'Balanced?':>12}")
print("-" * 62)
for v in COVARIATES:
    label    = COVARIATE_LABELS[v]
    balanced = "✓" if smd_post[v] < 0.10 else "✗"
    print(f"{label:<25} {smd_pre[v]:>10.3f} {smd_post[v]:>10.3f} {balanced:>12}")


# --- Figure 8: Love plot (SMD before vs. after) ------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
y_pos = np.arange(len(COVARIATES))
labels = [COVARIATE_LABELS[v] for v in COVARIATES]

ax.scatter([smd_pre[v]  for v in COVARIATES], y_pos, color=PALETTE["control"],
           zorder=3, s=80, label="Before PSM")
ax.scatter([smd_post[v] for v in COVARIATES], y_pos, color=PALETTE["treat"],
           zorder=3, s=80, marker="D", label="After PSM")

for i, v in enumerate(COVARIATES):
    ax.plot([smd_pre[v], smd_post[v]], [i, i], color="grey", lw=1, zorder=2)

ax.axvline(0.10, color="red", linestyle="--", linewidth=1, label="SMD = 0.10 threshold")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Standardized Mean Difference")
ax.set_title("Balance Improvement from Propensity Score Matching\n(Love Plot)", fontweight="bold")
ax.legend(fontsize=9)
sns.despine()
plt.tight_layout()
savefig("fig8_love_plot")


# =============================================================================
# 9. ATT ESTIMATION ON THE MATCHED SAMPLE
# =============================================================================

"""
We estimate the ATT in two ways:

(a) Unadjusted difference in means (simple ATT estimator):
    ATT_hat = E[reported | treated, matched] − E[reported | control, matched]

(b) Doubly-robust logit regression on the matched sample with full covariate
    adjustment. Bias-correction for any residual imbalance remaining
    after matching (Robins & Rotnitzky 1995; Ho et al. 2007).

Coefficients from the logistic regression are log-odds.  We compute Average
Marginal Effects (AMEs) for interpretability: the AME gives the average
change in the probability of reporting associated with a one-unit change
in the regressor.
"""

print("\n" + "=" * 60)
print("ATT ESTIMATION (matched sample)")
print("=" * 60)

# --- 9a. Unadjusted difference in means --------------------------------------

att_means = (
    matched_data.groupby("relationship_dummy")["reported_dummy"]
    .mean()
    .rename({0: "Did Not Know", 1: "Knew Aggressor"})
)
att_simple = att_means["Knew Aggressor"] - att_means["Did Not Know"]

print(f"\nUnadjusted reporting rates (matched sample):")
print(att_means.round(3).to_string())
print(f"\nSimple ATT estimate: {att_simple:+.3f} ({att_simple*100:+.1f} percentage points)")


# --- Figure 9: Reporting rates, matched sample --------------------------------

fig, ax = plt.subplots(figsize=(6, 4))
colors_bar = [PALETTE["control"], PALETTE["treat"]]
bars = ax.bar(att_means.index, att_means.values * 100, color=colors_bar,
              edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
ax.set_ylabel("Reporting Rate (%)")
ax.set_ylim(0, att_means.max() * 100 * 1.2)
ax.set_title("Reporting Rates by Victim–Offender Relationship\n(Matched Sample, ATT)", fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
sns.despine()
plt.tight_layout()
savefig("fig9_att_reporting_rates")


# --- 9b. Doubly-robust logit on matched sample --------------------------------

logit_matched_pooled = fit_logit(matched_data, "reported_dummy", REGRESSORS)

logit_matched_by_crime = {}
for ct in sorted(matched_data["newoff"].unique()):
    logit_matched_by_crime[ct] = fit_logit(
        matched_data[matched_data["newoff"] == ct], "reported_dummy", REGRESSORS
    )

model_names_matched = ["Pooled"] + [CRIME_LABELS_SHORT[ct] for ct in sorted(logit_matched_by_crime)]

matched_table = summary_col(
    [logit_matched_pooled] + [logit_matched_by_crime[ct] for ct in sorted(logit_matched_by_crime)],
    stars=True,
    model_names=model_names_matched,
    float_format="%0.3f",
    info_dict={"N": lambda x: f"{int(x.nobs):,}"},
)
print("\nLogistic regression on matched sample (log-odds coefficients):")
print(matched_table)


# --- 9c. Average Marginal Effects (AME) for the pooled matched model ---------

"""
AMEs are computed by:
  1. Predicting Pr(Y=1) for each observation with the actual X values.
  2. Predicting Pr(Y=1) for each observation with the variable of interest
     shifted by +1.
  3. Taking the sample average of the differences.
"""

def compute_ame(results, data: pd.DataFrame, var: str) -> float:
    """Compute the average marginal effect of a binary variable."""
    x_base = sm.add_constant(data[REGRESSORS], has_constant="add")
    x_alt  = x_base.copy()
    x_alt[var] = 1 - x_alt[var]      # flip the binary variable
    p_base = results.predict(x_base)
    p_alt  = results.predict(x_alt)
    return float((p_alt - p_base).mean())


print("\nAverage Marginal Effects (pooled matched logit):")
print(f"{'Variable':<25} {'AME':>8} {'Direction':>15}")
print("-" * 52)
for v in REGRESSORS:
    ame   = compute_ame(logit_matched_pooled, matched_data, v)
    label = COVARIATE_LABELS.get(v, v) if v != "relationship_dummy" else "Knew Aggressor (treatment)"
    direction = "↑ more likely to report" if ame > 0 else "↓ less likely to report"
    print(f"{label:<25} {ame:>+8.3f}   {direction}")


# --- 9d. AME for 'relationship_dummy' by crime type --------------------------

print("\nAME of 'Knew Aggressor' on reporting probability, by crime type:")
print(f"{'Crime Type':<25} {'AME':>8} {'N treated':>12}")
print("-" * 48)
for ct in sorted(logit_matched_by_crime):
    crime_matched = matched_data[matched_data["newoff"] == ct]
    ame = compute_ame(logit_matched_by_crime[ct], crime_matched, "relationship_dummy")
    n_t = int((crime_matched["relationship_dummy"] == 1).sum())
    label = CRIME_LABELS_SHORT[ct]
    print(f"{label:<25} {ame:>+8.3f} {n_t:>12,}")
    

# =============================================================================
# 10. REGRESSION COMPARISON: NAIVE vs. MATCHED
# =============================================================================
 
"""
We compare coefficients on relationship_dummy and the key covariates between
the naive (unmatched) and matched logistic regressions.
 
The focus is on the Rape/Sexual Assault column (crime type 1), where the
effect of selection bias is most pronounced.  I also highlight a sign
reversal and coefficient shifts across other covariates.
 
Reading the table
-----------------
All coefficients are log-odds.  A negative coefficient on relationship_dummy
means knowing one's aggressor reduces the log-odds of reporting.
We convert to odds ratios (exp(coef)) for interpretability and compute the
percentage change in odds: (OR - 1) * 100.
"""
 
print("\n" + "=" * 60)
print("REGRESSION COMPARISON: NAIVE vs. MATCHED")
print("=" * 60)
 
# Variables of interest and their display labels
FOCUS_VARS = {
    "relationship_dummy": "Knew Aggressor (treatment)",
    "married_dummy":      "Married",
    "minor_dummy":        "Minor (12-17)",
    "lower_income_dummy": "Low Income (<$35k)",
}
 
# Crime type to focus on
FOCUS_CRIME = 1  # Rape / Sexual Assault
 
# Retrieve the four models we need
naive_pooled   = logit_full          # from Section 6
naive_rape     = naive_by_crime[FOCUS_CRIME]
matched_pooled = logit_matched_pooled
matched_rape   = logit_matched_by_crime[FOCUS_CRIME]
 
rape_naive_data   = df[df["newoff"] == FOCUS_CRIME]
rape_matched_data = matched_data[matched_data["newoff"] == FOCUS_CRIME]
 
 
def coef_row(var, naive_res, matched_res, data_naive, data_matched):
    """
    For a given variable, extract naive and matched coefficients, compute
    odds ratio and AME for each, and return as a dict for display.
    """
    naive_coef   = naive_res.params[var]
    matched_coef = matched_res.params[var]
    naive_pval   = naive_res.pvalues[var]
    matched_pval = matched_res.pvalues[var]
 
    def stars(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
 
    naive_ame   = compute_ame(naive_res,   data_naive,   var)
    matched_ame = compute_ame(matched_res, data_matched, var)
 
    return {
        "naive_coef":   naive_coef,
        "naive_sig":    stars(naive_pval),
        "naive_or":     np.exp(naive_coef),
        "naive_ame":    naive_ame,
        "matched_coef": matched_coef,
        "matched_sig":  stars(matched_pval),
        "matched_or":   np.exp(matched_coef),
        "matched_ame":  matched_ame,
        "coef_change":  matched_coef - naive_coef,
        "ame_change":   matched_ame  - naive_ame,
    }
 
 
# --- 10a. Pooled comparison ---------------------------------------------------
 
print("\n--- Pooled model (all crime types) ---")
print(f"{'Variable':<30} {'Naive coef':>11} {'Matched coef':>13} {'Delta coef':>10}  "
      f"{'Naive AME':>10} {'Matched AME':>12} {'Delta AME':>10}")
print("-" * 105)
 
for var, label in FOCUS_VARS.items():
    r = coef_row(var, naive_pooled, matched_pooled, df, matched_data)
    print(f"{label:<30} "
          f"{r['naive_coef']:>+9.3f}{r['naive_sig']:<2} "
          f"{r['matched_coef']:>+11.3f}{r['matched_sig']:<2} "
          f"{r['coef_change']:>+9.3f}  "
          f"{r['naive_ame']:>+9.3f} "
          f"{r['matched_ame']:>+11.3f} "
          f"{r['ame_change']:>+10.3f}")
 
 
# --- 10b. Rape/Sexual Assault deep-dive ---------------------------------------
 
print(f"\n--- Rape / Sexual Assault (crime type {FOCUS_CRIME}) ---")
print(f"  Naive N = {int(naive_rape.nobs):,}   |   Matched N = {int(matched_rape.nobs):,}")
print()
print(f"{'Variable':<30} {'Naive coef':>11} {'Naive OR':>9} {'Naive AME':>10}  |  "
      f"{'Matched coef':>13} {'Matched OR':>11} {'Matched AME':>12}")
print("-" * 110)
 
for var, label in FOCUS_VARS.items():
    r = coef_row(var, naive_rape, matched_rape, rape_naive_data, rape_matched_data)
    print(f"{label:<30} "
          f"{r['naive_coef']:>+9.3f}{r['naive_sig']:<2} "
          f"{r['naive_or']:>8.3f}  "
          f"{r['naive_ame']:>+9.3f}   |  "
          f"{r['matched_coef']:>+11.3f}{r['matched_sig']:<2} "
          f"{r['matched_or']:>10.3f}  "
          f"{r['matched_ame']:>+11.3f}")
 
 
# --- 10c. Narrative ----------------------------------------------------------
 
rel = coef_row("relationship_dummy", naive_rape, matched_rape,
               rape_naive_data, rape_matched_data)
 
print(f"""
KEY FINDINGS - Rape / Sexual Assault
-------------------------------------
1. TREATMENT EFFECT (relationship_dummy)
   Naive   log-odds : {rel['naive_coef']:+.3f}  ->  OR = {rel['naive_or']:.3f}
           ({(rel['naive_or']-1)*100:+.1f}% change in odds of reporting)
           AME = {rel['naive_ame']:+.3f} ({rel['naive_ame']*100:+.1f} pp)
 
   Matched log-odds : {rel['matched_coef']:+.3f}  ->  OR = {rel['matched_or']:.3f}
           ({(rel['matched_or']-1)*100:+.1f}% change in odds of reporting)
           AME = {rel['matched_ame']:+.3f} ({rel['matched_ame']*100:+.1f} pp)
 
   The naive model underestimated the deterrent effect of knowing one's
   aggressor for sexual assault victims. After removing selection bias
   through matching, the estimated reduction in reporting probability is
   {abs(rel['ame_change'])*100:.1f} pp larger than the naive estimate.
   The matched OR of {rel['matched_or']:.3f} implies the odds of reporting are
   {abs((rel['matched_or']-1)*100):.1f}% lower for victims who knew their aggressor,
   compared to an observationally similar victim assaulted by a stranger.
""")
 
 
# --- 10d. Figure 10: Coefficient comparison across all crime types -----------
 
crime_types_sorted = sorted(naive_by_crime.keys())
 
naive_coefs, naive_cis     = [], []
matched_coefs, matched_cis = [], []
 
for ct in crime_types_sorted:
    var = "relationship_dummy"
    naive_coefs.append(naive_by_crime[ct].params[var])
    naive_cis.append(1.96 * naive_by_crime[ct].bse[var])
    matched_coefs.append(logit_matched_by_crime[ct].params[var])
    matched_cis.append(1.96 * logit_matched_by_crime[ct].bse[var])
 
labels = [CRIME_LABELS_SHORT[ct] for ct in crime_types_sorted]
x = np.arange(len(labels))
w = 0.3
 
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, naive_coefs,   w, label="Naive (unmatched)",
       color=PALETTE["control"], alpha=0.8, edgecolor="white")
ax.bar(x + w/2, matched_coefs, w, label="Matched (PSM)",
       color=PALETTE["treat"],   alpha=0.8, edgecolor="white")
ax.errorbar(x - w/2, naive_coefs,   yerr=naive_cis,
            fmt="none", color="black", capsize=4, linewidth=1)
ax.errorbar(x + w/2, matched_coefs, yerr=matched_cis,
            fmt="none", color="black", capsize=4, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Log-odds coefficient (+/- 95% CI)")
ax.set_title(
    "Effect of Knowing One's Aggressor on Reporting Probability\n"
    "Naive vs. PSM-Matched Logistic Regression, by Crime Type",
    fontweight="bold"
)
ax.legend()
sns.despine()
plt.tight_layout()
savefig("fig10_coef_comparison")
 
 
# --- 10e. Figure 11: Covariate coefficient shifts for Rape/Sexual Assault ----
 
focus_vars_list = list(FOCUS_VARS.keys())
focus_labels    = list(FOCUS_VARS.values())
 
naive_vals   = [naive_rape.params[v]       for v in focus_vars_list]
matched_vals = [matched_rape.params[v]     for v in focus_vars_list]
naive_ci     = [1.96 * naive_rape.bse[v]   for v in focus_vars_list]
matched_ci   = [1.96 * matched_rape.bse[v] for v in focus_vars_list]
 
x = np.arange(len(focus_vars_list))
 
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, naive_vals,   w, label="Naive (unmatched)",
       color=PALETTE["control"], alpha=0.8, edgecolor="white")
ax.bar(x + w/2, matched_vals, w, label="Matched (PSM)",
       color=PALETTE["treat"],   alpha=0.8, edgecolor="white")
ax.errorbar(x - w/2, naive_vals,   yerr=naive_ci,
            fmt="none", color="black", capsize=4, linewidth=1)
ax.errorbar(x + w/2, matched_vals, yerr=matched_ci,
            fmt="none", color="black", capsize=4, linewidth=1)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(focus_labels, rotation=15, ha="right")
ax.set_ylabel("Log-odds coefficient (+/- 95% CI)")
ax.set_title(
    "Coefficient Shift: Naive vs. Matched Regression\nRape / Sexual Assault",
    fontweight="bold"
)
ax.legend()
sns.despine()
plt.tight_layout()
savefig("fig11_rape_coef_shift")


# =============================================================================
# 11. SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY OF FINDINGS")
print("=" * 60)
print(f"""
Simple ATT (unadjusted matched difference):
  Victims who knew their aggressor were {abs(att_simple)*100:.1f} percentage
  points {'less' if att_simple < 0 else 'more'} likely to report to police,
  compared to a matched sample of victims who did not know their aggressor.

Key caveats:
  - The CIA (unconfoundedness) assumption cannot be tested. Unmeasured
    confounders — such as emotional dependence, fear of retaliation, or
    community norms — could bias these estimates.
  - PSM is unweighted; estimates reflect the ATT on the observed sample,
    not a population-representative ATE.
  - Matching quality should be inspected in the Love Plot (fig8).

See README.md for full interpretation, data sources, and references.
""")
