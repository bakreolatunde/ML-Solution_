"""
BUS 210 - Deliverable #2: preliminary analysis
Research question: can early indicators predict student dropout, and is the
model fair across income, gender and disability groups?

Dataset: student_dropout_risk_2026.csv (Kaggle, M. K. Thalla, CC0)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             roc_auc_score, accuracy_score, silhouette_score)

RANDOM_STATE = 42
DATA_PATH = "student_dropout_risk_2026.csv"
OUT = "."


# 1. Load and inspect
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
print("Dropout rate: {:.1%}".format(df["dropout"].mean()))
print("Students under 18:", int((df["age"] < 18).sum()))

print("Rows where passed + failed > enrolled:",
      int((df["units_passed"] + df["units_failed"] > df["units_enrolled"]).sum()))


# 2. Exploratory analysis and figures

numeric = df.select_dtypes(include="number").drop(columns=["person_id"])
corr_with_target = numeric.corr()["dropout"].drop("dropout").sort_values()
print("\nCorrelation with dropout (top 8 by absolute value):")
print(corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index).head(8).round(2))

sns.set_theme(style="whitegrid")

# Figure 1: dropout rate by family income
order = ["Low", "Middle", "High"]
rate_income = df.groupby("family_income_level")["dropout"].mean().reindex(order)
plt.figure(figsize=(5, 4))
sns.barplot(x=rate_income.index, y=rate_income.values, color="#4C72B0")
plt.ylabel("Dropout rate"); plt.xlabel("Family income level")
plt.title("Dropout rate by family income")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_dropout_by_income.png", dpi=150); plt.close()

# Figure 2: GPA (semester 1) by outcome
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="gpa_semester_1", hue="dropout", bins=30,
             element="step", stat="density", common_norm=False)
plt.title("Semester 1 GPA by outcome (1 = dropout)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_gpa_by_outcome.png", dpi=150); plt.close()

# Figure 3: correlation heatmap of the numeric variables
plt.figure(figsize=(11, 9))
sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.7}, linewidths=0.2)
plt.title("Correlation heatmap (numeric variables)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_correlation_heatmap.png", dpi=150); plt.close()

# Figure 4: dropout rate by attendance band
df["attendance_band"] = pd.cut(df["attendance_rate"], bins=[-0.1, 50, 70, 85, 100],
                               labels=["0-50%", "50-70%", "70-85%", "85-100%"])
rate_att = df.groupby("attendance_band", observed=True)["dropout"].mean()
plt.figure(figsize=(5.5, 4))
sns.barplot(x=rate_att.index.astype(str), y=rate_att.values, color="#55A868")
plt.ylabel("Dropout rate"); plt.xlabel("Attendance rate")
plt.title("Dropout rate by attendance band")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_dropout_by_attendance.png", dpi=150); plt.close()
df = df.drop(columns="attendance_band")


# 3. Modeling
# Two feature sets:
#  FULL: everything except the ID
#  EARLY: excludes variables that are only known late (semester 2 GPA and the unit counts), to avoid look-ahead leakage in an early-warning system
target = "dropout"
drop_always = ["person_id", target]
late_vars = ["gpa_semester_2", "units_enrolled", "units_passed", "units_failed"]

def build_xy(exclude):
    X = df.drop(columns=drop_always + exclude)
    return X, df[target]

def make_preprocessor(X):
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ])

def get_models():
    return {
        "Logistic regression": LogisticRegression(max_iter=2000),
        "Decision tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                                random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                n_jobs=-1, random_state=RANDOM_STATE),
    }

results = []
fitted = {}
for label, exclude in [("FULL", []), ("EARLY", late_vars)]:
    X, y = build_xy(exclude)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    for name, model in get_models().items():
        pipe = Pipeline([("prep", make_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        results.append({
            "Feature set": label, "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "AUC": roc_auc_score(y_test, proba),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
        fitted[(label, name)] = (pipe, X_test, y_test, proba)

res = pd.DataFrame(results)
print("\nModel comparison (test set, cutoff 0.5):")
print(res.round(3).to_string(index=False))
res.round(4).to_csv(f"{OUT}/model_results.csv", index=False)


# 4. Interpretation: random forest feature importance (EARLY set)
pipe, X_test, y_test, proba = fitted[("EARLY", "Random forest")]
names = pipe.named_steps["prep"].get_feature_names_out()
imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
imp = imp.sort_values(ascending=False).head(10)
imp.index = [i.replace("num__", "").replace("cat__", "") for i in imp.index]
print("\nTop 10 features (random forest, EARLY set):")
print(imp.round(3))

plt.figure(figsize=(7, 4.5))
sns.barplot(x=imp.values, y=imp.index, color="#C44E52")
plt.xlabel("Importance"); plt.title("Top 10 features (random forest, early set)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_feature_importance.png", dpi=150); plt.close()

# Logistic regression coefficients (EARLY set), easier to explain
pipe_lr = fitted[("EARLY", "Logistic regression")][0]
coefs = pd.Series(pipe_lr.named_steps["model"].coef_[0],
                  index=pipe_lr.named_steps["prep"].get_feature_names_out())
coefs.index = [i.replace("num__", "").replace("cat__", "") for i in coefs.index]
print("\nLargest logistic regression coefficients (standardised, EARLY set):")
print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8).round(2))


# 5. Fairness check (random forest, EARLY set, cutoff 0.5)
pipe, X_test, y_test, proba = fitted[("EARLY", "Random forest")]
audit = X_test.copy()
audit["y"] = y_test.values
audit["pred"] = (proba >= 0.5).astype(int)

def group_metrics(g):
    pos = g[g["y"] == 1]; neg = g[g["y"] == 0]
    return pd.Series({
        "n": len(g),
        "actual dropout rate": g["y"].mean(),
        "predicted dropout rate": g["pred"].mean(),
        "recall (true positive rate)": (pos["pred"] == 1).mean(),
        "false positive rate": (neg["pred"] == 1).mean(),
    })

fair_tables = []
for col in ["gender", "family_income_level", "disability_status"]:
    t = audit.groupby(col).apply(group_metrics, include_groups=False)
    t.insert(0, "attribute", col)
    fair_tables.append(t)
fair = pd.concat(fair_tables)
print("\nFairness audit (random forest, EARLY set):")
print(fair.round(3).to_string())
fair.round(4).to_csv(f"{OUT}/fairness_audit.csv")


# 6. Optional: k-means risk profiles
cluster_vars: list[str] = ["gpa_semester_1", "attendance_rate", "assignments_submitted_rate",
                "missed_deadlines", "lms_logins_per_week", "stress_level"]
Z = StandardScaler().fit_transform(df[cluster_vars])
sample = np.random.RandomState(RANDOM_STATE).choice(len(Z), 10000, replace=False)
print("\nSilhouette scores (10,000-row sample):")
for k in range(2, 6):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    print(f"  k={k}: {silhouette_score(Z[sample], km.labels_[sample]):.3f}")

K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(Z)
df["cluster"] = km.labels_
profile = df.groupby("cluster")[cluster_vars + ["dropout"]].mean()
profile.insert(0, "size", df["cluster"].value_counts().sort_index())
print(f"\nCluster profiles (k={K}):")
print(profile.round(2).to_string())
profile.round(3).to_csv(f"{OUT}/cluster_profiles.csv")

print("\nDone. Figures and CSV tables saved.")
