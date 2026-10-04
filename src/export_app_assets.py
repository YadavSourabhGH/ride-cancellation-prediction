"""Export small result files that the Streamlit app uses for its explanations.

Run after src/train.py:
    python src/export_app_assets.py

It rebuilds the exact same train/test split as train.py (same features, same
random_state=42), scores the hold-out test set with the saved Random Forest and
a Logistic Regression baseline, and writes:
    report/test_predictions.csv   - true label + predicted probability per test row
    report/feature_importance.csv - permutation importance of each input column
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / 'data' / 'uber_request_data_raw.csv')

# Same feature preparation as src/train.py, so the split is identical.
df = raw.copy()
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df['request_timestamp'] = pd.to_datetime(df['request_timestamp'], errors='coerce')
df['target_cancelled'] = df['status'].eq('Cancelled').astype(int)
df['hour'] = df['request_timestamp'].dt.hour
df['day_of_week'] = df['request_timestamp'].dt.day_name()
df['month'] = df['request_timestamp'].dt.month
df['is_weekend'] = (df['request_timestamp'].dt.dayofweek >= 5).astype(int)

features = ['pickup_point', 'hour', 'day_of_week', 'month', 'is_weekend']
X, y = df[features], df['target_cancelled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

rf = joblib.load(ROOT / 'models' / 'ride_cancellation_model.joblib')
lr = Pipeline([('preprocess', clone(rf.named_steps['preprocess'])),
               ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
lr.fit(X_train, y_train)

pd.DataFrame({'y_true': y_test.values,
              'random_forest': rf.predict_proba(X_test)[:, 1],
              'logistic_regression': lr.predict_proba(X_test)[:, 1]}).to_csv(
    ROOT / 'report' / 'test_predictions.csv', index=False)

pi = permutation_importance(rf, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42)
pd.DataFrame({'feature': features, 'importance': pi.importances_mean, 'std': pi.importances_std}) \
    .sort_values('importance', ascending=False) \
    .to_csv(ROOT / 'report' / 'feature_importance.csv', index=False)
print('Wrote report/test_predictions.csv and report/feature_importance.csv')
