from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'
MODEL = ROOT / 'models'
FIG.mkdir(exist_ok=True)
MODEL.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
raw = pd.read_csv(DATA / 'uber_request_data_raw.csv')
df = raw.copy()
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df['request_timestamp'] = pd.to_datetime(df['request_timestamp'], errors='coerce')
df['drop_timestamp'] = pd.to_datetime(df['drop_timestamp'], errors='coerce')
# The prediction point is the request moment. Driver and drop-time fields are excluded to avoid leakage.
df['target_cancelled'] = (df['status'].eq('Cancelled')).astype(int)
df['hour'] = df['request_timestamp'].dt.hour
df['day_of_week'] = df['request_timestamp'].dt.day_name()
df['day_num'] = df['request_timestamp'].dt.dayofweek
df['month'] = df['request_timestamp'].dt.month
df['is_weekend'] = (df['day_num'] >= 5).astype(int)
df['time_band'] = pd.cut(df['hour'], bins=[-1, 5, 11, 16, 20, 23], labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late evening'])

# Save prepared data with fields that are useful for audit and downstream app development.
prepared_cols = ['request_id','pickup_point','status','request_timestamp','target_cancelled','hour','day_of_week','month','is_weekend','time_band']
df[prepared_cols].to_csv(DATA / 'uber_request_data_prepared.csv', index=False)

# EDA figures.
status_counts = df['status'].value_counts().reindex(['Trip Completed','Cancelled','No Cars Available']).dropna()
plt.figure(figsize=(9,5))
ax = status_counts.plot(kind='bar', color=['#2f6f6d','#d95f59','#e6a23c'])
ax.set_title('Request outcomes in the source data')
ax.set_xlabel('Outcome'); ax.set_ylabel('Requests'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIG/'outcome_counts.png', dpi=180); plt.close()

cancel_rate = df.groupby(['pickup_point','hour'], observed=True)['target_cancelled'].mean().reset_index()
pivot = cancel_rate.pivot(index='hour', columns='pickup_point', values='target_cancelled')
plt.figure(figsize=(10,5)); sns.lineplot(data=cancel_rate, x='hour', y='target_cancelled', hue='pickup_point', marker='o', palette=['#2f6f6d','#d95f59'])
plt.title('Driver-cancellation rate by request hour'); plt.ylabel('Driver-cancel rate'); plt.xlabel('Hour of day'); plt.ylim(0, 1); plt.tight_layout(); plt.savefig(FIG/'cancel_rate_by_hour.png', dpi=180); plt.close()

plt.figure(figsize=(9,5)); sns.countplot(data=df, x='hour', hue='status', hue_order=['Trip Completed','Cancelled','No Cars Available'], palette=['#2f6f6d','#d95f59','#e6a23c'])
plt.title('Request outcomes by hour'); plt.xlabel('Hour of day'); plt.ylabel('Requests'); plt.legend(title='Outcome', fontsize=9); plt.tight_layout(); plt.savefig(FIG/'outcomes_by_hour.png', dpi=180); plt.close()

# Modeling data. Only fields observable at request time are used.
features = ['pickup_point','hour','day_of_week','month','is_weekend']
X = df[features].copy(); y = df['target_cancelled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
cat = ['pickup_point','day_of_week']; num = ['hour','month','is_weekend']
pre = ColumnTransformer([('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat),
                         ('num', Pipeline([('impute', SimpleImputer(strategy='median')),('scale', StandardScaler())]), num)])
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=350, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1)
}
results=[]; fitted={}
for name, clf in models.items():
    pipe = Pipeline([('preprocess', pre), ('model', clf)])
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:,1]; pred = (p >= 0.5).astype(int)
    results.append({'model':name, 'roc_auc':roc_auc_score(y_test,p), 'average_precision':average_precision_score(y_test,p), 'precision':precision_score(y_test,pred,zero_division=0), 'recall':recall_score(y_test,pred,zero_division=0), 'f1':f1_score(y_test,pred,zero_division=0), 'accuracy':accuracy_score(y_test,pred)})
    fitted[name]=(pipe,p,pred)

metrics = pd.DataFrame(results).sort_values(['average_precision','roc_auc'], ascending=False)
metrics.to_csv(ROOT/'report/model_metrics.csv', index=False)
best_name = metrics.iloc[0]['model']; best_pipe, best_p, best_pred = fitted[best_name]
joblib.dump(best_pipe, MODEL/'ride_cancellation_model.joblib')

# Evaluation figures for the selected model.
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5.5,4.5)); sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False, xticklabels=['No driver cancel','Driver cancel'], yticklabels=['No driver cancel','Driver cancel'])
plt.title(f'Confusion matrix: {best_name.replace("_"," ").title()}'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.savefig(FIG/'confusion_matrix.png', dpi=180); plt.close()
fig, ax = plt.subplots(1,2, figsize=(12,5)); RocCurveDisplay.from_predictions(y_test,best_p,ax=ax[0],name=best_name); PrecisionRecallDisplay.from_predictions(y_test,best_p,ax=ax[1],name=best_name); ax[0].set_title('ROC curve'); ax[1].set_title('Precision-recall curve'); plt.tight_layout(); plt.savefig(FIG/'evaluation_curves.png', dpi=180); plt.close()

summary = {
    'raw_rows': int(len(raw)), 'raw_columns': int(raw.shape[1]), 'duplicate_rows': int(raw.duplicated().sum()),
    'missing_values': {k:int(v) for k,v in raw.isna().sum().items()},
    'status_counts': {str(k):int(v) for k,v in raw['Status'].value_counts().items()},
    'driver_cancel_rate': float(y.mean()), 'completed_rate': float((df['status']=='Trip Completed').mean()),
    'no_car_rate': float((df['status']=='No Cars Available').mean()),
    'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)), 'selected_model': best_name,
    'metrics': metrics.to_dict(orient='records'),
    'test_classification_report': classification_report(y_test,best_pred,output_dict=True,zero_division=0)
}
(ROOT/'report'/'analysis_summary.json').write_text(json.dumps(summary, indent=2, default=float))
(ROOT/'report'/'classification_report.txt').write_text(classification_report(y_test,best_pred, target_names=['No driver cancellation','Driver cancellation'], zero_division=0))
print(json.dumps(summary, indent=2, default=float))
