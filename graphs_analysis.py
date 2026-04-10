import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import joblib

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 300, "savefig.bbox": "tight"})

ANALYSIS_END = pd.Timestamp("2026-01-31")
CSV_PATH = "/content/master_diploma_ra_dataset_noisy.csv"
MODEL_PATH = "/content/active_auditor_rf_robust.pkl"

print("Loading model and preparing data...")
rf = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=["paid_at","due_date","ready_at","issued_date","cancelled_at","delivered_at","start_date","end_date"], low_memory=False)

fe = df.copy()
fe["days_to_analysis_end"] = (ANALYSIS_END - fe["due_date"]).dt.days.fillna(0)
fe["same_period_invoice_count"] = fe.groupby(["customer_id", "start_date", "end_date"])["order_id"].transform("count")
fe["is_delivered_no_pay"] = ((fe["status"] == "delivered") & fe["paid_at"].isna()).astype(int)
fe["overdue_days"] = np.where(fe["paid_at"].isna() & fe["due_date"].notna(), (ANALYSIS_END - fe["due_date"]).dt.days, 0)
fe["is_paid"] = (fe["status"] == "paid").astype(int)
fe["is_bad_debt"] = (fe["status"] == "bad debt").astype(int)
fe["is_cancelled"] = (fe["status"] == "cancelled").astype(int)
fe["has_no_payment"] = fe["paid_at"].isna().astype(int)
fe["delta_issue_to_start"] = (fe["issued_date"] - fe["start_date"]).dt.days
fe["is_late_issued"] = (fe["delta_issue_to_start"] >= 30).astype(int)
fe["is_overdue_90d"] = ((fe["days_to_analysis_end"] > 90) & fe["paid_at"].isna()).astype(int)
fe["delta_due_to_issue"] = (fe["due_date"] - fe["issued_date"]).dt.days
fe["delta_ready_to_issue"] = (fe["issued_date"] - fe["ready_at"]).dt.total_seconds() / 3600
fe["service_period_days"] = (fe["end_date"] - fe["start_date"]).dt.days
fe["paid_vs_due_delta"] = np.where(fe["paid_at"].notna() & fe["due_date"].notna(), (fe["paid_at"] - fe["due_date"]).dt.days, 999)

cust_baseline = fe.groupby("customer_id")["subtotal"].median().rename("cust_baseline")
fe = fe.merge(cust_baseline, on="customer_id", how="left")
fe["subtotal_deviation"] = ((fe["subtotal"] - fe["cust_baseline"]).abs() / fe["cust_baseline"].clip(lower=1))

fe["issue_day_of_month"] = fe["issued_date"].dt.day.fillna(0).astype(int)
fe["issue_month"] = fe["issued_date"].dt.month.fillna(0).astype(int)
fe["issue_quarter"] = fe["issued_date"].dt.quarter.fillna(0).astype(int)
fe["billing_cycle_months"] = ((fe["end_date"] - fe["start_date"]).dt.days / 30.44).round().clip(1, 12).fillna(1).astype(int)
fe["curr_usd"] = (fe["currency_code"] == "USD").astype(int)
fe["curr_eur"] = (fe["currency_code"] == "EUR").astype(int)
fe["curr_uah"] = (fe["currency_code"] == "UAH").astype(int)

cust_agg = fe.groupby("customer_id").agg(cust_invoice_count=("order_id", "count"), cust_bad_debt_rate=("is_bad_debt", "mean")).reset_index()
fe = fe.merge(cust_agg, on="customer_id", how="left")
fe = fe.sort_values(["customer_id", "issued_date"]).reset_index(drop=True)
cust_modal_cycle = fe.groupby("customer_id")["billing_cycle_months"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 1).rename("modal_cycle")
fe = fe.merge(cust_modal_cycle, on="customer_id", how="left")
fe["prev_end_date"] = fe.groupby("customer_id")["end_date"].shift(1)
fe["gap_from_prev"] = (fe["start_date"] - fe["prev_end_date"]).dt.days.fillna(0)
fe["is_after_large_gap"] = (fe["gap_from_prev"] > fe["modal_cycle"] * 45).astype(int)

FEATURE_COLS = [
    "subtotal", "total", "exchange_rate", "local_currency_total",
    "delta_issue_to_start", "delta_due_to_issue", "delta_ready_to_issue",
    "service_period_days", "days_to_analysis_end", "paid_vs_due_delta",
    "is_paid", "has_no_payment", "is_overdue_90d", "is_late_issued",
    "subtotal_deviation", "issue_day_of_month", "issue_month", "issue_quarter",
    "billing_cycle_months", "curr_usd", "curr_eur", "curr_uah",
    "cust_invoice_count", "cust_bad_debt_rate", "gap_from_prev", "is_after_large_gap",
    "same_period_invoice_count", "is_delivered_no_pay", "overdue_days"
]

TARGET = "true_leakage_label"
df_model = fe.dropna(subset=["issued_date"]).sort_values("issued_date").copy()
df_model = df_model[df_model[TARGET] != "cancelled"].reset_index(drop=True)

for col in FEATURE_COLS:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

split_idx = int(len(df_model) * 0.80)
df_test = df_model.iloc[split_idx:].copy()

X_test = df_test[FEATURE_COLS].values
y_test = df_test[TARGET].values

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)
classes = rf.classes_

print("Generating visualizations...")

fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df_test, x=TARGET, order=df_test[TARGET].value_counts().index, palette="viridis", ax=ax)
ax.set_yscale("log")
ax.set_title("Test Set Class Distribution (Log Scale)", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of Invoices (Log)", fontsize=12)
ax.set_xlabel("Leakage Type", fontsize=12)
plt.xticks(rotation=45, ha="right")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig("thesis_fig1_class_distribution.png")
plt.close()

y_test_bin = label_binarize(y_test, classes=classes)
fig, ax = plt.subplots(figsize=(10, 7))

for i, class_name in enumerate(classes):
    if class_name == "clean": continue 
    
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
    
    ax.plot(recall, precision, lw=2, label=f"{class_name} (AP = {ap:.2f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves for Leakage Types", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("thesis_fig2_pr_curves.png")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(data=df_test[df_test[TARGET].isin(["clean", "wrong_sum"])], 
            x="subtotal_deviation", hue=TARGET, fill=True, common_norm=False, palette=["#2ecc71", "#e74c3c"], ax=axes[0])
axes[0].set_xlim(-0.05, 0.25)
axes[0].set_title("Density: Subtotal Deviation\n(Clean vs. Wrong Sum)", fontsize=12)
axes[0].set_xlabel("Subtotal Deviation Ratio")

sns.kdeplot(data=df_test[df_test[TARGET].isin(["clean", "late_issuance"])], 
            x="delta_issue_to_start", hue=TARGET, fill=True, common_norm=False, palette=["#2ecc71", "#3498db"], ax=axes[1])
axes[1].set_xlim(-10, 60)
axes[1].set_title("Density: Days from Start to Issuance\n(Clean vs. Late Issuance)", fontsize=12)
axes[1].set_xlabel("Delta (Days)")

plt.tight_layout()
plt.savefig("thesis_fig3_feature_densities.png")
plt.close()

df_test["y_pred"] = y_pred
anomalies = [c for c in classes if c != "clean"]

tp_counts = []
fn_counts = []

for cls in anomalies:
    mask = df_test[TARGET] == cls
    tp = (df_test.loc[mask, "y_pred"] == cls).sum()
    fn = mask.sum() - tp
    tp_counts.append(tp)
    fn_counts.append(fn)

x = np.arange(len(anomalies))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, tp_counts, width, label='True Positives (Detected)', color='#2ecc71')
bars2 = ax.bar(x + width/2, fn_counts, width, label='False Negatives (Missed)', color='#e74c3c')

ax.set_ylabel('Number of Invoices', fontsize=12)
ax.set_title('Detection Success Rate by Leakage Type', fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(anomalies, rotation=30, ha="right")
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("thesis_fig4_error_analysis.png")
plt.close()