# Does Knowing One's Aggressor Reduce Reporting to Authorities?
### A Causal Inference Analysis of the National Crime Victimization Survey (1993–2023)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: BJS NCVS](https://img.shields.io/badge/Data-BJS%20NCVS-green)](https://bjs.ojp.gov/national-crime-victimization-survey-ncvs-api)

---

## Research Question

> **Does having a prior relationship with one's aggressor causally reduce the probability of reporting the crime to the police?**

Victims of interpersonal violence are consistently less likely to report their victimization to law enforcement than victims of stranger violence. Yet the extent to which this gap reflects a *causal* effect of the victim–offender relationship, rather than compositional differences in the types of crimes, victims, or circumstances involved, remains underexplored in the empirical literature.

This project uses **Propensity Score Matching (PSM)** on 30 years of the **National Crime Victimization Survey (NCVS)** to estimate the **Average Treatment Effect on the Treated (ATT)**: the reduction in reporting probability attributable to knowing one's aggressor, for victims who did know their aggressor.

---

## Data

| Item | Detail |
|---|---|
| **Source** | Bureau of Justice Statistics (BJS) — NCVS Select Personal Victimization |
| **API endpoint** | `https://api.ojp.gov/bjsdataset/v1/gcuy-rt5g.json` |
| **Years** | 1993–2023 |
| **Raw records** | 67,779 incidents |
| **Working sample** | 54,648 (after listwise deletion of missing values) |
| **Unit of observation** | Individual victimization incident |
| **Survey weight** | `wgtviccy` (used in descriptive figures only (see [Note on Weights](#note-on-survey-weights))) |

### Variables

| Variable | Source column | Definition |
|---|---|---|
| `reported_dummy` | `notify` | **Outcome**: 1 = reported to police; 0 = did not report |
| `relationship_dummy` | `direl` | **Treatment**: 1 = knew aggressor (intimate/relative/acquaintance); 0 = stranger |
| `female_dummy` | `sex` | 1 = female victim |
| `married_dummy` | `marital` | 1 = currently married |
| `minor_dummy` | `ager` | 1 = victim aged 12–17 |
| `lower_income_dummy` | `hincome1` | 1 = household income < $35,000 |
| `newoff` | `newoff` | Crime type (1 = Rape/Sexual Assault, 2 = Robbery, 3 = Aggravated Assault, 4 = Simple Assault, 5 = Personal Theft) |

### Missing Values

| Variable | Missing | Share |
|---|---|---|
| `lower_income_dummy` | 7,260 | 10.7% |
| `relationship_dummy` | 5,555 | 8.2% |
| `reported_dummy` | 1,095 | 1.6% |
| `married_dummy` | 270 | 0.4% |

Missingness in `relationship_dummy` arises when the victim could not identify the number or identity of offenders, these cases are excluded as they are not classifiable as either treatment or control.

---

## Descriptive Overview

Simple Assault dominates the sample (~63%), followed by Robbery and Aggravated Assault. Rape and Sexual Assault represent roughly 5% of incidents, a figure that is itself likely a substantial undercount given well-documented under-reporting.

![Crime composition](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig1_crime_composition.png)

Victim–offender relationships vary substantially by crime type. Sexual assault and simple assault show the highest share of known aggressors, while robbery is predominantly committed by strangers.

![Relationship by crime](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig3_relationship_by_crime.png)

Overall reporting rates are low across all crime types, with Aggravated Assault showing the highest reporting rate and Personal Theft the lowest.

![Reporting rate](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig4_reporting_rate.png)

Among unreported crimes specifically, the share involving a known aggressor is markedly higher for sexual assault, motivating the core research question.

![Unreported relationship](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig5_unreported_relationship.png)

---

## Methodology

### Identification Strategy

We estimate a causal effect using **Propensity Score Matching**. The identifying assumptions are:

1. **Conditional Independence Assumption (CIA / Unconfoundedness)**
   Conditional on the observed covariates $(X)$, treatment assignment (knowing one's aggressor) is independent of potential outcomes:
   $$Y(0), Y(1) \perp D \mid X$$
   This is a strong assumption. Unmeasured confounders, emotional dependency, fear of retaliation, social norms, prior victimization history, may still bias the estimates.

2. **Common Support**
   For every propensity score value in the treated group, there exists a comparable control unit:
   $$0 < P(D = 1 \mid X) < 1$$
   Verified graphically (see [Common Support](#common-support) below).

3. **SUTVA (Stable Unit Treatment Value Assumption)**
   One unit's reporting decision does not affect another unit's outcome.

### Target Parameter: ATT vs. ATE

We target the **Average Treatment Effect on the Treated (ATT)**, not the population ATE:
$$\text{ATT} = E[Y(1) - Y(0) \mid D = 1]$$

The ATT answers: *"For victims who knew their aggressor, how much did that relationship reduce their probability of reporting?"* This is the policy-relevant quantity when designing outreach for victims of interpersonal violence.

### PSM Procedure

1. **Propensity score estimation**: logistic regression of treatment $(D)$ on covariates $(X)$.
2. **Stratified 1:1 nearest-neighbour matching** within crime-type strata (`newoff`). Stratifying by crime type removes it as a confound, since victim–offender relationship distributions vary substantially across crime categories.
3. **Balance diagnostics**: Standardized Mean Differences (SMD) before and after matching. Rule of thumb: SMD < 0.10 indicates adequate balance (Stuart 2010).
4. **Outcome estimation**: doubly-robust logistic regression on the matched sample, with full covariate adjustment.
5. **Average Marginal Effects (AME)** computed from the matched-sample logit for interpretable probability-scale estimates.

### Note on Survey Weights

Survey weights (`wgtviccy`) are applied **only for the descriptive figures**. They are intentionally excluded from the PSM procedure because PSM targets the ATT on the observed sample, not a population-representative ATE, and because evidence on including survey weights in propensity score models is mixed (Austin, Jembere & Chiu 2016; Lenis et al. 2019).

---

## Results

### Pre-Matching Balance

Before matching, the treated (knew aggressor) and control (stranger) groups differ significantly on all observed covariates:

| Covariate | Did Not Know | Knew Aggressor | Difference | Z-stat | Sig. |
|---|---|---|---|---|---|
| Married | 0.325 | 0.227 | −0.098 | −25.54 | *** |
| Female | 0.367 | 0.603 | +0.236 | +55.16 | *** |
| Minor (12–17) | 0.137 | 0.214 | +0.077 | +23.38 | *** |
| Low Income (<$35k) | 0.459 | 0.551 | +0.092 | +21.55 | *** |

The treated group is disproportionately female, younger, lower-income, and less likely to be married, all characteristics that are themselves associated with reporting behaviour. A naive comparison of reporting rates would conflate the effect of the victim–offender relationship with these demographic differences, directly motivating the PSM approach.

![Pre-matching means](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig6_prebalance_means.png)

### Common Support

The propensity score distributions of treated and control units overlap substantially, satisfying the common support assumption.

![Common support](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig7_common_support.png)

### Post-Matching Balance

After 1:1 nearest-neighbour matching within crime-type strata, all SMDs fall to 0.000, well below the 0.10 threshold, confirming near-perfect balance on all observed covariates.

| Covariate | SMD Pre | SMD Post | Balanced |
|---|---|---|---|
| Married | 0.219 | 0.000 | ✓ |
| Female | 0.486 | 0.000 | ✓ |
| Minor (12–17) | 0.202 | 0.000 | ✓ |
| Low Income (<$35k) | 0.185 | 0.000 | ✓ |

![Love plot](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig8_love_plot.png)

### Treatment Effect: How Matching Changes the Estimates

The figure below compares the `relationship_dummy` coefficient (effect of knowing one's aggressor on reporting log-odds) across all crime types, before and after matching. The shift between naive and matched estimates is itself evidence that selection bias was present in the unmatched sample and has been corrected for.

![Coefficient comparison](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig10_coef_comparison.png)

### Focus: Rape and Sexual Assault

The most policy-relevant and econometrically instructive results are in the rape/sexual assault stratum.

| | Naive (unmatched) | Matched (PSM) |
|---|---|---|
| **Log-odds (relationship_dummy)** | −0.293*** | −0.942*** |
| **Odds Ratio** | 0.746 | 0.390 |
| **Change in odds of reporting** | −25.4% | −61.0% |
| **N** | 2,962 | 4,124 |

The naive model already identified a significant negative effect, but **matching reveals the true magnitude to be three times larger**. The matched estimate implies that for a sexual assault victim who knew her aggressor, the odds of reporting to police are **61% lower** than for an observationally comparable victim assaulted by a stranger. Selection bias, specifically the overrepresentation of female, younger, and lower-income victims in the treatment group, was attenuating the apparent effect in the naive regression.

The figure below shows how each covariate's coefficient shifts between the naive and matched regressions for the sexual assault stratum:

![Rape coefficient shift](https://raw.githubusercontent.com/clementhennet/personal-victimization/refs/heads/main/fig11_rape_coef_shift.png)

Two additional covariate findings warrant attention:

**Minor (12–17)**: the coefficient increases from +0.786 (naive) to +1.563 (matched). Once compositional confounds are removed, minor victims of sexual assault are substantially *more* likely to report, likely because reporting is often initiated by a parent, guardian, or school official rather than the victim herself.

**Low Income**: the coefficient reverses sign, from +0.145 (naive, not significant) to −1.628 (matched, p<0.001). In the unmatched sample, lower-income victims are disproportionately in the treatment group and in crime categories with lower baseline reporting rates, creating an upward bias. Once those confounds are removed by matching, lower-income sexual assault victims are revealed to be substantially *less* likely to report, consistent with the secondary victimization literature (greater distrust of police, economic dependence on the aggressor, limited access to victim services).

> **Caution**: These estimates rest on the CIA (unconfoundedness) assumption, which cannot be tested. Unmeasured confounders, emotional dependency, fear of retaliation, community norms, may still bias the estimates. Results should be interpreted as *conditional associations after removing observable selection bias*, not as purely causal effects.

---

## Repository Structure

```
ncvs_project/
├── analysis.py              # Main analysis script (end-to-end)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── victimization_data.csv   # Auto-generated cache (first run downloads from API)
└── outputs/
    └── figures/
        ├── fig1_crime_composition.png
        ├── fig2_crime_trends.png
        ├── fig3_relationship_by_crime.png
        ├── fig4_reporting_rate.png
        ├── fig5_unreported_relationship.png
        ├── fig6_prebalance_means.png
        ├── fig7_common_support.png
        ├── fig8_love_plot.png
        ├── fig9_att_reporting_rates.png
        ├── fig10_coef_comparison.png
        └── fig11_rape_coef_shift.png
```

---

## Running the Analysis

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run

```bash
python analysis.py
```

On first run, the script downloads ~30 years of NCVS data from the BJS API and caches it locally as `victimization_data.csv`. Subsequent runs use the cache. All figures are saved to `outputs/figures/`.

### Manual data download (if the API is unavailable)

If the automatic download fails (e.g., institutional network restrictions):

1. Visit the [BJS NCVS API page](https://bjs.ojp.gov/national-crime-victimization-survey-ncvs-api)
2. Query the API directly at `https://api.ojp.gov/bjsdataset/v1/gcuy-rt5g.json`
3. Save the result as `victimization_data.csv` in the project root.

The script detects the cached file and skips the API call automatically.

---

## References

- Austin, P. C., Jembere, N., & Chiu, M. (2016). Propensity score matching and complex surveys. *Statistical Methods in Medical Research*, 27(4), 1240–1257.
- Caliendo, M., & Kopeinig, S. (2008). Some practical guidance for the implementation of propensity score matching. *Journal of Economic Surveys*, 22(1), 31–72.
- Ho, D. E., Imai, K., King, G., & Stuart, E. A. (2007). Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. *Political Analysis*, 15(3), 199–236.
- Lenis, D., Nguyen, T. Q., Dong, N., & Stuart, E. A. (2019). It's all about balance: propensity score matching in the context of complex survey data. *Biostatistics*, 20(1), 147–163.
- Robins, J. M., & Rotnitzky, A. (1995). Semiparametric efficiency in multivariate regression models with missing data. *Journal of the American Statistical Association*, 90(429), 122–129.
- Stuart, E. A. (2010). Matching methods for causal inference: a review and a look forward. *Statistical Science*, 25(1), 1–21.

---

## License

MIT
