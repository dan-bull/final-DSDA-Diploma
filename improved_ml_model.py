import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

SEED = 42
np.random.seed(SEED)

ANALYSIS_END = pd.Timestamp("2026-01-31")
CSV_PATH = "master_diploma_ra_dataset_noisy.csv"

date_cols = ["paid_at","due_date","ready_at","issued_date","cancelled_at","delivered_at","start_date","end_date"]
df = pd.read_csv(CSV_PATH, parse_dates=date_cols, low_memory=False)

fe = df.copy()

fe["delta_issue_to_start"] = (fe["issued_date"] - fe["start_date"]).dt.days
fe["delta_due_to_issue"] = (fe["due_date"] - fe["issued_date"]).dt.days
fe["delta_ready_to_issue"] = (fe["issued_date"] - fe["ready_at"]).dt.total_seconds() / 3600
fe["service_period_days"] = (fe["end_date"] - fe["start_date"]).dt.days
fe["days_to_analysis_end"] = (ANALYSIS_END - fe["due_date"]).dt.days.fillna(0)
fe["paid_vs_due_delta"] = np.where(fe["paid_at"].notna() & fe["due_date"].notna(), (fe["paid_at"] - fe["due_date"]).dt.days, 999)

fe["same_period_invoice_count"] = fe.groupby(["customer_id", "start_date", "end_date"])["order_id"].transform("count")
fe["is_delivered_no_pay"] = ((fe["status"] == "delivered") & fe["paid_at"].isna()).astype(int)
fe["overdue_days"] = np.where(fe["paid_at"].isna() & fe["due_date"].notna(), (ANALYSIS_END - fe["due_date"]).dt.days, 0)

fe["is_paid"] = (fe["status"] == "paid").astype(int)
fe["is_bad_debt"] = (fe["status"] == "bad debt").astype(int)
fe["is_cancelled"] = (fe["status"] == "cancelled").astype(int)
fe["has_no_payment"] = fe["paid_at"].isna().astype(int)
fe["is_overdue_90d"] = ((fe["days_to_analysis_end"] > 90) & fe["paid_at"].isna()).astype(int)
fe["is_late_issued"] = (fe["delta_issue_to_start"] >= 30).astype(int)

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

FEATURE_COLS_IMPROVED = [
    "subtotal", "total", "exchange_rate", "local_currency_total",
    "delta_issue_to_start", "delta_due_to_issue", "delta_ready_to_issue",
    "service_period_days", "days_to_analysis_end", "paid_vs_due_delta",
    "is_paid", "has_no_payment", "is_overdue_90d", "is_late_issued",
    "subtotal_deviation",
    "issue_day_of_month", "issue_month", "issue_quarter",
    "billing_cycle_months",
    "curr_usd", "curr_eur", "curr_uah",
    "cust_invoice_count", "cust_bad_debt_rate",
    "gap_from_prev", "is_after_large_gap",
    "same_period_invoice_count",
    "is_delivered_no_pay",
    "overdue_days"
]

TARGET = "true_leakage_label"

df_model = fe.dropna(subset=["issued_date"]).sort_values("issued_date").copy()
df_model = df_model[df_model[TARGET] != "cancelled"].reset_index(drop=True)

for col in FEATURE_COLS_IMPROVED:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

split_idx = int(len(df_model) * 0.80)
df_train  = df_model.iloc[:split_idx]
df_test   = df_model.iloc[split_idx:]

X_train = df_train[FEATURE_COLS_IMPROVED].values
y_train = df_train[TARGET].values
X_test  = df_test[FEATURE_COLS_IMPROVED].values
y_test  = df_test[TARGET].values

rf = RandomForestClassifier(
    n_estimators=500, max_depth=None,
    min_samples_split=5, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced",
    n_jobs=-1, random_state=SEED,
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred, digits=4, zero_division=0))

joblib.dump(rf, "active_auditor_rf_robust.pkl")

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_).plot(ax=ax, cmap="Blues", xticks_rotation=45)
ax.set_title("Confusion Matrix (Robust Model)")
plt.tight_layout()
plt.savefig("fig_robust_confusion_matrix.png")
plt.close()

feat_imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS_IMPROVED).sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(11, 7))
bars = feat_imp.plot(kind='barh', ax=ax, color="#3498db", edgecolor="white", width=0.75)
ax.set_title("Top 20 Feature Importances (Robust Model)", fontsize=12)
ax.set_xlabel("Mean Decrease in Gini Impurity")
plt.tight_layout()
plt.savefig("fig_robust_feature_importance.png")
plt.close()