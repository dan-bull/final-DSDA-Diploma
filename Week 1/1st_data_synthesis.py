# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

np.random.seed(42)
NUM_CUSTOMERS = 3500
START_ANALYSIS = datetime(2024, 12, 1)
END_ANALYSIS = datetime(2026, 1, 31)
CHURN_TARGET = 600
VAT_RATE = 0.20
CURRENCIES = {'USD': 1.0, 'EUR': 0.92, 'UAH': 41.0}

customers = []
for i in range(1, NUM_CUSTOMERS + 1):
    start_offset = np.random.randint(-12, 1)
    cust_start = START_ANALYSIS + relativedelta(months=start_offset)
    billing_cycle = np.random.randint(1, 13)

    is_churned = i <= CHURN_TARGET
    churn_date = None
    if is_churned:
        days_range = (END_ANALYSIS - START_ANALYSIS).days
        churn_date = START_ANALYSIS + timedelta(days=np.random.randint(30, days_range))

    customers.append({
        'customer_id': i,
        'cust_start': cust_start,
        'billing_cycle': billing_cycle,
        'churn_date': churn_date,
        'base_subtotal': np.random.uniform(500, 5000) # B2B pricing
    })

all_invoices = []
order_counter = 50000

for cust in customers:
    current_date = cust['cust_start']

    while current_date <= (END_ANALYSIS + relativedelta(months=2)):
        # standard churn stop, unless we force a Zombie error
        force_zombie = (cust['churn_date'] is not None and
                        current_date > cust['churn_date'] and
                        np.random.rand() < 0.02) # 2% chance of zombie billing

        if cust['churn_date'] and current_date > cust['churn_date'] and not force_zombie:
            break

        period_start = current_date
        period_end = current_date + relativedelta(months=cust['billing_cycle']) - timedelta(days=1)

        error_type = np.random.choice([
            'none', 'missing_invoice', 'gap', 'uncollected', 'wrong_sum', 'duplicate', 'late_issuance'
        ], p=[0.75, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03])

        if force_zombie: error_type = 'zombie'
        if error_type == 'missing_invoice':
            current_date += relativedelta(months=cust['billing_cycle'])
            continue
        if error_type == 'gap':
            current_date += relativedelta(months=cust['billing_cycle'] * 2)
            continue

        subtotal = cust['base_subtotal']
        if error_type == 'wrong_sum': subtotal *= np.random.choice([0.9, 1.1])

        total = subtotal * (1 + VAT_RATE)
        curr = np.random.choice(list(CURRENCIES.keys()))
        ex_rate = CURRENCIES[curr]

        issued_date = period_start + timedelta(days=np.random.randint(0, 5))
        if error_type == 'late_issuance':
            issued_date = period_start + timedelta(days=np.random.randint(30, 45))

        due_date = issued_date + timedelta(days=30)

        status = 'delivered'
        paid_at = pd.NaT
        cancelled_at = pd.NaT

        if error_type == 'uncollected':
            if (END_ANALYSIS - due_date).days > 90:
                status = 'bad debt'
            else:
                status = 'delivered'
        elif np.random.rand() < 0.05:
            status = 'cancelled'
            cancelled_at = issued_date + timedelta(days=2)
        else:
            status = 'paid'
            paid_at = due_date + timedelta(days=np.random.randint(-5, 20))

        inv_data = {
            'customer_id': cust['customer_id'],
            'total': round(total, 2),
            'paid_at': paid_at,
            'due_date': due_date,
            'order_id': order_counter,
            'products': "B2B Subscription Services",
            'ready_at': issued_date - timedelta(hours=5),
            'subtotal': round(subtotal, 2),
            'vat_rate': VAT_RATE,
            'issued_date': issued_date,
            'cancelled_at': cancelled_at,
            'delivered_at': issued_date,
            'status': status,
            'currency_code': curr,
            'exchange_rate': ex_rate,
            'local_currency_total': round(total * ex_rate, 2),
            'start_date': period_start,
            'end_date': period_end
        }

        all_invoices.append(inv_data)

        if error_type == 'duplicate':
            dup_inv = inv_data.copy()
            order_counter += 1
            dup_inv['order_id'] = order_counter
            all_invoices.append(dup_inv)

        order_counter += 1
        current_date += relativedelta(months=cust['billing_cycle'])

df = pd.DataFrame(all_invoices)
df.to_csv('master_diploma_ra_dataset.csv', index=False)
print(f"Dataset ready: {len(df)} invoices generated.")
