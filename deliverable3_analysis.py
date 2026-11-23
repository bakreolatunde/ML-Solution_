"""
BUS 210 - Deliverable #3: extended analysis for the final paper
Research question: can early indicators predict student dropout, and is the
model fair across income, gender and disability groups?

Builds on deliverable2_analysis.py and adds:
  1. Train / validation / test split (60 / 20 / 20), so cutoffs are chosen on
     validation data and reported on the untouched test set
  2. 5-fold cross-validation of the three models
  3. Cutoff sweep, precision-recall trade-off and calibration check
  4. Bootstrap 95% confidence intervals for every group metric
  5. Formal fairness gaps (demographic parity, equal opportunity, equalised odds)
  6. Test of removing income (and its likely proxies) as predictors
  7. Exploratory group-specific cutoffs and an income x disability check

Dataset: student_dropout_risk_2026.csv (Kaggle, M. K. Thalla, CC0)
Run:  python deliverable3_analysis.py   (CSV must be in the same folder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

RS = 42
DATA_PATH = "student_dropout_risk_2026.csv"
OUT = "d3_outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
rng = np.random.RandomState(RS)

# --------------------------------------------------------------------------
# 0. Data and split
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
target = "dropout"
late_vars = ["gpa_semester_2", "units_enrolled", "units_passed", "units_failed"]
X_all = df.drop(columns=["person_id", target] + late_vars)   # EARLY feature set
y_all = df[target]

# 60% train / 20% validation / 20% test, stratified on the target
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=RS)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=RS)
print(f"Train {len(X_train)}, validation {len(X_val)}, test {len(X_test)}")


def make_pipe(X, model):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
    return Pipeline([("prep", prep), ("model", model)])


def lr():
    return LogisticRegression(max_iter=2000)


def rates(y, p):
    """Confusion-matrix based rates for arrays of true labels and 0/1 predictions."""
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return {
        "n": len(y), "actual_rate": y.mean(), "flagged_share": p.mean(),
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# --------------------------------------------------------------------------
# 1. Cross-validation (train + validation data, early features)
# --------------------------------------------------------------------------
print("\n=== 1. 5-fold cross-validation (AUC) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
cv_models = {
    "Logistic regression": lr(),
    "Decision tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=RS),
    "Random forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                            n_jobs=-1, random_state=RS)}
cv_rows = []
for name, m in cv_models.items():
    scores = cross_val_score(make_pipe(X_tmp, m), X_tmp, y_tmp, cv=cv, scoring="roc_auc")
    cv_rows.append({"Model": name, "Mean AUC": scores.mean(), "SD": scores.std()})
cv_res = pd.DataFrame(cv_rows)
print(cv_res.round(4).to_string(index=False))
cv_res.round(4).to_csv(f"{OUT}/cv_results.csv", index=False)


# 2. Main model: logistic regression on the early feature se
pipe = make_pipe(X_train, lr()).fit(X_train, y_train)
p_val = pipe.predict_proba(X_val)[:, 1]
p_test = pipe.predict_proba(X_test)[:, 1]
print("\n=== 2. Logistic regression (early set) ===")
print(f"Validation AUC {roc_auc_score(y_val, p_val):.3f} | test AUC {roc_auc_score(y_test, p_test):.3f}")
print(f"Test Brier score {brier_score_loss(y_test, p_test):.3f}")


# 3. Cutoff sweep on VALIDATION data
print("\n=== 3. Cutoff sweep (validation data) ===")
grid = np.round(np.arange(0.05, 0.81, 0.05), 2)
sweep = []
for t in grid:
    r = rates(y_val, (p_val >= t).astype(int))
    sweep.append({"cutoff": t, "flagged_share": r["flagged_share"],
                  "precision": r["precision"], "recall": r["recall"], "fpr": r["fpr"]})
sweep = pd.DataFrame(sweep)
print(sweep.round(3).to_string(index=False))
sweep.round(4).to_csv(f"{OUT}/cutoff_sweep_validation.csv", index=False)

# Candidate cutoffs, all chosen on validation data only
fine = np.round(np.arange(0.01, 0.99, 0.01), 2)
rec_fine = np.array([rates(y_val, (p_val >= t).astype(int))["recall"] for t in fine])


def cutoff_for_recall(target_recall):
    ok = fine[rec_fine >= target_recall]
    return float(ok.max()) if len(ok) else float(fine.min())


candidates = {
    "Default 0.50": 0.50,
    "Cost-based, miss 3x worse (1/(1+3))": 0.25,
    "Cost-based, miss 4x worse (1/(1+4))": 0.20,
    "Recall target 80%": cutoff_for_recall(0.80),
    "Recall target 85%": cutoff_for_recall(0.85),
    "Capacity: flag top 30%": float(np.quantile(p_val, 0.70)),
}
rows = []
for label, t in candidates.items():
    r = rates(y_test, (p_test >= t).astype(int))
    rows.append({"Rule": label, "cutoff": round(t, 3), **{k: r[k] for k in
                 ["flagged_share", "precision", "recall", "fpr", "TP", "FP", "FN"]}})
cut_test = pd.DataFrame(rows)
print("\nCandidate cutoffs evaluated on the TEST set:")
print(cut_test.round(3).to_string(index=False))
cut_test.round(4).to_csv(f"{OUT}/cutoff_candidates_test.csv", index=False)

T_STAR = candidates["Recall target 80%"]   # operating cutoff used below
print(f"\nOperating cutoff (80% recall target on validation data): {T_STAR:.2f}")

# Figure 6: precision, recall and FPR against the cutoff (validation data)
plt.figure(figsize=(6.5, 4.2))
plt.plot(sweep["cutoff"], sweep["recall"], marker="o", label="Recall")
plt.plot(sweep["cutoff"], sweep["precision"], marker="s", label="Precision")
plt.plot(sweep["cutoff"], sweep["fpr"], marker="^", label="False positive rate")
plt.axvline(T_STAR, color="grey", linestyle="--", label=f"Chosen cutoff ({T_STAR:.2f})")
plt.xlabel("Cutoff"); plt.ylabel("Rate"); plt.title("Cutoff sweep (validation data)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/fig6_cutoff_sweep.png", dpi=150); plt.close()

# --------------------------------------------------------------------------
# 4. Fairness audit with bootstrap confidence intervals (test set)
# --------------------------------------------------------------------------
print("\n=== 4. Fairness audit at the chosen cutoff, with 95% bootstrap CIs ===")
B = 1000
audit = X_test[["gender", "family_income_level", "disability_status"]].copy()
audit["y"] = y_test.values
audit["pred"] = (p_test >= T_STAR).astype(int)
audit["score"] = p_test
order = {"gender": ["Female", "Male", "Other"],
         "family_income_level": ["Low", "Middle", "High"],
         "disability_status": [0, 1]}


def boot(sub):
    """Bootstrap recall and FPR for one group. Returns arrays of B draws."""
    y = sub["y"].values; p = sub["pred"].values
    n = len(y); rec = np.empty(B); fpr = np.empty(B)
    for b in range(B):
        i = rng.randint(0, n, n)
        yy, pp = y[i], p[i]
        rec[b] = ((yy == 1) & (pp == 1)).sum() / max((yy == 1).sum(), 1)
        fpr[b] = ((yy == 0) & (pp == 1)).sum() / max((yy == 0).sum(), 1)
    return rec, fpr


boots = {}
group_rows = []
for attr, levels in order.items():
    for lv in levels:
        sub = audit[audit[attr] == lv]
        r = rates(sub["y"], sub["pred"])
        rec_b, fpr_b = boot(sub)
        boots[(attr, lv)] = (rec_b, fpr_b)
        group_rows.append({
            "attribute": attr, "group": lv, "n": r["n"], "actual_rate": r["actual_rate"],
            "flagged_share": r["flagged_share"], "precision": r["precision"],
            "recall": r["recall"],
            "recall_CI": f"{np.percentile(rec_b, 2.5):.3f}-{np.percentile(rec_b, 97.5):.3f}",
            "fpr": r["fpr"],
            "fpr_CI": f"{np.percentile(fpr_b, 2.5):.3f}-{np.percentile(fpr_b, 97.5):.3f}"})
group_tbl = pd.DataFrame(group_rows)
print(group_tbl.round(3).to_string(index=False))
group_tbl.round(4).to_csv(f"{OUT}/fairness_groups_with_CI.csv", index=False)

# Pairwise gaps with confidence intervals
print("\nGaps between groups (difference in rates, 95% bootstrap CI):")
pairs = [("family_income_level", "Low", "High"), ("family_income_level", "Low", "Middle"),
         ("disability_status", 1, 0), ("gender", "Female", "Male")]
gap_rows = []
for attr, a, b_ in pairs:
    for name, k in [("recall", 0), ("fpr", 1)]:
        d = boots[(attr, a)][k] - boots[(attr, b_)][k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        gap_rows.append({"comparison": f"{attr}: {a} minus {b_}", "metric": name,
                         "gap": d.mean(), "CI_low": lo, "CI_high": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
gap_tbl = pd.DataFrame(gap_rows)
print(gap_tbl.round(3).to_string(index=False))
gap_tbl.round(4).to_csv(f"{OUT}/fairness_gaps_CI.csv", index=False)

# Formal fairness metrics (largest gap between any two groups of an attribute)
print("\nFormal fairness gaps (max minus min across groups):")
formal = []
for label, cutoff in [("Default 0.50", 0.50), (f"Chosen {T_STAR:.2f}", T_STAR)]:
    pred = (p_test >= cutoff).astype(int)
    for attr, levels in order.items():
        rs = [rates(y_test[audit[attr].values == lv], pred[audit[attr].values == lv]) for lv in levels]
        sel = [r["flagged_share"] for r in rs]; rec = [r["recall"] for r in rs]
        fpr = [r["fpr"] for r in rs]; prec = [r["precision"] for r in rs]
        formal.append({"cutoff": label, "attribute": attr,
                       "demographic_parity_gap": max(sel) - min(sel),
                       "equal_opportunity_gap (recall)": max(rec) - min(rec),
                       "fpr_gap": max(fpr) - min(fpr),
                       "equalised_odds_gap": max(max(rec) - min(rec), max(fpr) - min(fpr)),
                       "precision_gap": max(prec) - min(prec)})
formal = pd.DataFrame(formal)
print(formal.round(3).to_string(index=False))
formal.round(4).to_csv(f"{OUT}/fairness_formal_metrics.csv", index=False)

# Figure 7: recall and FPR by group with 95% CIs
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
nice = {"gender": "Gender", "family_income_level": "Income", "disability_status": "Disability"}
labels = [f"{nice[a]}: {('Yes' if g == 1 else 'No') if a == 'disability_status' else g}"
          for a, g in zip(group_tbl["attribute"], group_tbl["group"])]
for ax, (k, name) in zip(axes, [(0, "Recall"), (1, "False positive rate")]):
    vals = group_tbl["recall" if k == 0 else "fpr"].values
    lo = [np.percentile(boots[(a, g)][k], 2.5) for a, g in zip(group_tbl["attribute"], group_tbl["group"])]
    hi = [np.percentile(boots[(a, g)][k], 97.5) for a, g in zip(group_tbl["attribute"], group_tbl["group"])]
    ax.errorbar(vals, range(len(vals)), xerr=[vals - np.array(lo), np.array(hi) - vals],
                fmt="o", capsize=3)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels)
    ax.set_xlabel(name); ax.set_title(f"{name} by group (95% CI)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_group_metrics_CI.png", dpi=150); plt.close()

# --------------------------------------------------------------------------
# 5. ROC and calibration by income group
# --------------------------------------------------------------------------
print("\n=== 5. Calibration and ROC by income group (test set) ===")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for lv in order["family_income_level"]:
    m = (audit["family_income_level"] == lv).values
    fpr_c, tpr_c, _ = roc_curve(y_test[m], p_test[m])
    auc = roc_auc_score(y_test[m], p_test[m])
    axes[0].plot(fpr_c, tpr_c, label=f"{lv} (AUC {auc:.3f})")
    frac_pos, mean_pred = calibration_curve(y_test[m], p_test[m], n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, marker="o", label=lv)
    print(f"{lv}: AUC {auc:.3f}, mean predicted {p_test[m].mean():.3f}, actual {y_test[m].mean():.3f}, "
          f"Brier {brier_score_loss(y_test[m], p_test[m]):.3f}")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curve by income group"); axes[0].legend()
axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[1].set_xlabel("Mean predicted risk"); axes[1].set_ylabel("Observed dropout rate")
axes[1].set_title("Calibration by income group"); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_roc_calibration_by_income.png", dpi=150); plt.close()

# --------------------------------------------------------------------------
# 6. Does removing income (and likely proxies) close the gap?
# --------------------------------------------------------------------------
print("\n=== 6. Removing income and proxies (logistic regression) ===")
variants = {
    "Baseline (all early features)": [],
    "Without family income": ["family_income_level"],
    "Without income, tuition status, scholarship": ["family_income_level", "tuition_up_to_date",
                                                    "scholarship_holder"],
}
var_rows = []
for label, drop in variants.items():
    Xtr, Xv, Xte = X_train.drop(columns=drop), X_val.drop(columns=drop), X_test.drop(columns=drop)
    pp = make_pipe(Xtr, lr()).fit(Xtr, y_train)
    pv, pt = pp.predict_proba(Xv)[:, 1], pp.predict_proba(Xte)[:, 1]
    rec_v = np.array([rates(y_val, (pv >= t).astype(int))["recall"] for t in fine])
    ok = fine[rec_v >= 0.80]
    t_var = float(ok.max()) if len(ok) else 0.01          # same 80% recall rule on validation
    pred = (pt >= t_var).astype(int)
    inc = X_test["family_income_level"].values
    lo = rates(y_test[inc == "Low"], pred[inc == "Low"])
    hi = rates(y_test[inc == "High"], pred[inc == "High"])
    var_rows.append({"Model": label, "test AUC": roc_auc_score(y_test, pt), "cutoff": t_var,
                     "overall recall": rates(y_test, pred)["recall"],
                     "overall FPR": rates(y_test, pred)["fpr"],
                     "recall Low": lo["recall"], "recall High": hi["recall"],
                     "recall gap (Low-High)": lo["recall"] - hi["recall"],
                     "FPR Low": lo["fpr"], "FPR High": hi["fpr"],
                     "FPR gap (Low-High)": lo["fpr"] - hi["fpr"]})
var_tbl = pd.DataFrame(var_rows)
print(var_tbl.round(3).T.to_string(header=False))
var_tbl.round(4).to_csv(f"{OUT}/remove_income_test.csv", index=False)

# --------------------------------------------------------------------------
# 7. Exploratory: group-specific cutoffs by income level
# --------------------------------------------------------------------------
print("\n=== 7. Exploratory: income-specific cutoffs (target 80% recall in each group) ===")
inc_val = X_val["family_income_level"].values
inc_test = X_test["family_income_level"].values
pred_group = np.zeros(len(y_test), dtype=int)
spec_rows = []
for lv in order["family_income_level"]:
    mv = inc_val == lv
    rec_g = np.array([rates(y_val[mv], (p_val[mv] >= t).astype(int))["recall"] for t in fine])
    ok = fine[rec_g >= 0.80]
    t_g = float(ok.max()) if len(ok) else 0.01
    mt = inc_test == lv
    pred_group[mt] = (p_test[mt] >= t_g).astype(int)
    r_single = rates(y_test[mt], (p_test[mt] >= T_STAR).astype(int))
    r_group = rates(y_test[mt], pred_group[mt])
    spec_rows.append({"group": lv, "group cutoff": t_g,
                      "recall (single cutoff)": r_single["recall"], "FPR (single cutoff)": r_single["fpr"],
                      "recall (group cutoffs)": r_group["recall"], "FPR (group cutoffs)": r_group["fpr"]})
spec = pd.DataFrame(spec_rows)
print(spec.round(3).to_string(index=False))
r_all_single = rates(y_test, (p_test >= T_STAR).astype(int))
r_all_group = rates(y_test, pred_group)
print(f"Overall: single cutoff recall {r_all_single['recall']:.3f}, FPR {r_all_single['fpr']:.3f} | "
      f"group cutoffs recall {r_all_group['recall']:.3f}, FPR {r_all_group['fpr']:.3f}")
spec.round(4).to_csv(f"{OUT}/group_specific_cutoffs.csv", index=False)

# --------------------------------------------------------------------------
# 8. Intersection: income x disability (chosen cutoff, test set)
# --------------------------------------------------------------------------
print("\n=== 8. Income x disability at the chosen cutoff ===")
inter = []
for lv in order["family_income_level"]:
    for d in [0, 1]:
        m = (audit["family_income_level"] == lv) & (audit["disability_status"] == d)
        r = rates(audit.loc[m, "y"], audit.loc[m, "pred"])
        inter.append({"income": lv, "disability": d, "n": r["n"], "actual_rate": r["actual_rate"],
                      "recall": r["recall"], "fpr": r["fpr"]})
inter = pd.DataFrame(inter)
print(inter.round(3).to_string(index=False))
inter.round(4).to_csv(f"{OUT}/intersections_income_disability.csv", index=False)

print(f"\nDone. Tables and figures saved in ./{OUT}/")
