from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import uuid
from datetime import datetime

app = Flask(__name__)

# Path to your CSV data store
CSV_PATH = r"E:\PCCOE\Semesters\6th\ML\Sagar ML Mini Project\data\fraud_transactions_data.csv"

# Load existing transaction numbers
df = pd.read_csv(CSV_PATH)
seen_txn_numbers = set(df['TransactionNumber'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    txn_number = request.form.get('TransactionNumber', '').strip()
    try:
        amount = float(request.form.get('TransactionAmount'))
    except ValueError:
        return jsonify({'error': 'Invalid transaction amount.'})

    # --- New logic ---
    if txn_number not in seen_txn_numbers:
        # First time seeing this transaction number:
        # 1) Mark it legitimate
        # 2) Append it to the CSV for future duplicate checks
        seen_txn_numbers.add(txn_number)
        new_row = {
            'TransactionID':     str(uuid.uuid4())[:8],
            'TransactionNumber': txn_number,
            'TransactionAmount': amount,
            'TransactionTime':   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'IsFraud':           0
        }
        # Append without writing header
        pd.DataFrame([new_row]).to_csv(CSV_PATH, mode='a', index=False, header=False)

        return jsonify({
            'result': 'Legitimate (secure)',
            'probability': None
        })

    else:
        # Second (or more) time seeing this transaction number:
        # flag as fraud
        return jsonify({
            'result': 'Fraud (Not secure)',
            'probability': None
        })

if __name__ == '__main__':
    app.run(debug=True)
