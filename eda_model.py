import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# --- Configuration ---
SEED = 42
ANALYSIS_END = pd.Timestamp("2026-01-31")
PALETTE = {
    "clean": "#27ae60", "uncollected": "#e74c3c", "duplicate": "#e67e22",
    "late_issuance": "#3498db", "wrong_sum": "#9b59b6", "zombie_billing": "#1abc9c", "cancelled": "#95a5a6",
}
sns.set_theme(style="whitegrid", font_scale=1.1)

# --- Data Loading ---
df = pd.read_csv("/content/master_diploma_ra_dataset.csv", parse_dates=[
    "paid_at", "due_date", "ready_at", "issued_date", "cancelled_at", "delivered_at", "start_date", "end_date"
])

# --- Feature Engineering ---
df_fe = df.copy()

# Temporal Deltas
df_fe["delta_issue_to_start"] = (df_fe["issued_date"] - df_fe["start_date"]).dt.days
df_fe["delta_due_to_issue"]   = (df_fe["due_date"] - df_fe["issued_date"]).dt.days
df_fe["delta_ready_to_issue"] = (df_fe["issued_date"] - df_fe["ready_at"]).dt.total_seconds() / 3600
df_fe["service_period_days"]  = (df_fe["end_date"] - df_fe["start_date"]).dt.days
df_fe["days_to_analysis_end"] = (ANALYSIS_END - df_fe["due_date"]).dt.days.fillna(0)
df_fe["paid_vs_due_delta"]    = np.where(df_fe["paid_at"].notna(), (df_fe["paid_at"] - df_fe["due_date"]).dt.days, 999)

# Risk Flags & Customer Aggregates
df_fe["is_bad_debt"]      = (df_fe["status"] == "bad debt").astype(int)
df_fe["is_late_issued"]   = (df_fe["delta_issue_to_start"] >= 30).astype(int)
df_fe["billing_cycle_months"] = ((df_fe["end_date"] - df_fe["start_date"]).dt.days / 30.44).round().clip(1, 12).astype("Int64")

cust_stats = df_fe.groupby("customer_id").agg(
    cust_baseline=("subtotal", "median"),
    cust_invoice_count=("order_id", "count"),
    cust_bad_debt_n=("is_bad_debt", "sum"),
    modal_cycle=("billing_cycle_months", lambda x: x.mode().iloc[0] if not x.mode().empty else 1)
).reset_index()

df_fe = df_fe.merge(cust_stats, on="customer_id", how="left")
df_fe["subtotal_deviation"] = (df_fe["subtotal"] - df_fe["cust_baseline"]).abs() / df_fe["cust_baseline"].clip(lower=1)
df_fe["cust_bad_debt_rate"] = df_fe["cust_bad_debt_n"] / df_fe["cust_invoice_count"].clip(lower=1)

# Zombie & Duplicate Detection
df_fe = df_fe.sort_values(["customer_id", "issued_date"])
df_fe["gap_from_prev"] = (df_fe["start_date"] - df_fe.groupby("customer_id")["end_date"].shift(1)).dt.days.fillna(0)
df_fe["is_after_large_gap"] = (df_fe["gap_from_prev"] > df_fe["modal_cycle"] * 45).astype(int)

dup_mask = df_fe.duplicated(subset=["customer_id", "subtotal", "start_date", "end_date"], keep=False)

# --- Label Reconstruction ---
def assign_label(row):
    if row["status"] == "bad debt": return "uncollected"
    if row["status"] == "cancelled": return "cancelled"
    if dup_mask.loc[row.name]: return "duplicate"
    if row["is_late_issued"]: return "late_issuance"
    if row["subtotal_deviation"] > 0.07: return "wrong_sum"
    if row["is_after_large_gap"] and row["status"] == "paid": return "zombie_billing"
    return "clean"

df_fe["leakage_label"] = df_fe.apply(assign_label, axis=1)

# --- Modeling ---
FEATURE_COLS = [
    "subtotal", "total", "exchange_rate", "local_currency_total", "delta_issue_to_start", 
    "delta_due_to_issue", "delta_ready_to_issue", "service_period_days", "days_to_analysis_end", 
    "paid_vs_due_delta", "subtotal_deviation", "billing_cycle_months", "cust_invoice_count", 
    "cust_bad_debt_rate", "gap_from_prev", "is_after_large_gap"
]

df_model = df_fe.dropna(subset=["issued_date"]).sort_values("issued_date")
df_model = df_model[df_model["leakage_label"] != "cancelled"].reset_index(drop=True)

for col in FEATURE_COLS:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

split_idx = int(len(df_model) * 0.8)
train, test = df_model.iloc[:split_idx], df_model.iloc[split_idx:]

rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED)
rf.fit(train[FEATURE_COLS], train["leakage_label"])

# --- Evaluation & Impact ---
y_pred = rf.predict(test[FEATURE_COLS])
print(classification_report(test["leakage_label"], y_pred))

EBITDA_MARGIN = 0.25
test["y_pred"] = y_pred
tp_mask = (test["y_pred"] == test["leakage_label"]) & (test["leakage_label"] != "clean")

recovered_usd = test.loc[tp_mask, "subtotal"].sum()
total_leak_usd = test.loc[test["leakage_label"] != "clean", "subtotal"].sum()

print(f"Total Leakage: ${total_leak_usd:,.2f}")
print(f"EBITDA Recovered: ${recovered_usd * EBITDA_MARGIN:,.2f}")
print(f"Detection Rate: {(recovered_usd/total_leak_usd)*100:.1f}%")

joblib.dump(rf, "active_auditor_rf_model.pkl")