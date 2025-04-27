import os
import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# Absolute base directory
BASE_DIR = r"E:\PCCOE\Semesters\6th\ML\Sagar ML Mini Project"
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(DATA_DIR, 'fraud_transactions_data.csv')

# Configuration
num_rows = 1200
fraud_ratio = 0.15
num_frauds = int(num_rows * fraud_ratio)

# Helper: random timestamp
def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

start_date = datetime(2025, 4, 1)
end_date   = datetime(2025, 4, 10)

# Generate transaction numbers in the range TXN100–TXN100000
transaction_numbers = [
    f"TXN{random.randint(100, 100000)}"
    for _ in range(num_rows - num_frauds)
]
fraudulent_numbers = random.sample(transaction_numbers, num_frauds)

data = []
# Non-fraud entries
for txn_number in transaction_numbers:
    data.append({
        "TransactionID":       str(uuid.uuid4())[:8],
        "TransactionNumber":   txn_number,
        "TransactionAmount":   round(random.uniform(100, 5000), 2),
        "TransactionTime":     random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S"),
        "IsFraud":             0
    })

# Fraud entries (duplicate transaction numbers)
for txn_number in fraudulent_numbers:
    data.append({
        "TransactionID":       str(uuid.uuid4())[:8],
        "TransactionNumber":   txn_number,
        "TransactionAmount":   round(random.uniform(100, 5000), 2),
        "TransactionTime":     random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S"),
        "IsFraud":             1
    })

# Shuffle and save
random.shuffle(data)
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Generated {OUTPUT_CSV} with {len(df)} rows.")
