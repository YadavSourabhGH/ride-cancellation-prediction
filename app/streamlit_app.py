"""Ride Cancellation Prediction - Case Study 66.

An explanatory Streamlit app: it predicts the cancellation risk of a ride request
and explains, step by step, how the data, the pipeline and the model produce that number.

Run:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'ride_cancellation_model.joblib'
RAW_PATH = ROOT / 'data' / 'uber_request_data_raw.csv'
METRICS_PATH = ROOT / 'report' / 'model_metrics.csv'
PRED_PATH = ROOT / 'report' / 'test_predictions.csv'
IMPORTANCE_PATH = ROOT / 'report' / 'feature_importance.csv'
FIG = ROOT / 'figures'

THRESHOLD = 0.50
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
TEAL, RED, AMBER, GREY = '#2f6f6d', '#d95f59', '#e6a23c', '#8a94a3'
PICKUP_COLORS = alt.Scale(domain=['Airport', 'City'], range=[TEAL, RED])
OUTCOME_ORDER = ['Trip Completed', 'Cancelled', 'No Cars Available']
OUTCOME_COLORS = alt.Scale(domain=OUTCOME_ORDER, range=[TEAL, RED, AMBER])

st.set_page_config(page_title='Ride Cancellation Prediction', page_icon='🚕', layout='wide')


# ----------------------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_raw():
    """Raw data for the explanations. Both date formats in the file are read here
    (e.g. '11/7/2016 11:51' and '13-07-2016 08:18:13'), so every request gets an hour."""
    raw = pd.read_csv(RAW_PATH)
    df = raw.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df['request_time'] = pd.to_datetime(df['request_timestamp'], format='mixed', dayfirst=True)
    df['hour'] = df['request_time'].dt.hour
    df['target_cancelled'] = df['status'].eq('Cancelled').astype(int)
    return raw, df


@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


model = load_model()
raw, data = load_raw()
metrics_df = load_csv(METRICS_PATH)
preds = load_csv(PRED_PATH)
importance = load_csv(IMPORTANCE_PATH)
BASE_RATE = data['target_cancelled'].mean()


def make_row(pickup, hour, day, month):
    return pd.DataFrame([{'pickup_point': pickup, 'hour': hour, 'day_of_week': day,
                          'month': month, 'is_weekend': int(day in ['Saturday', 'Sunday'])}])


def predict(pickup, hour, day, month):
    return float(model.predict_proba(make_row(pickup, hour, day, month))[0, 1])


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header('Describe a ride request')
    st.caption('These are the only things the app knows at the moment a customer books a ride.')
    pickup_point = st.radio('Pickup point', ['City', 'Airport'], horizontal=True,
                            help='Where the ride starts. City rides go to the airport, Airport rides go to the city.')
    hour = st.slider('Request hour (24-hour clock)', 0, 23, 7,
                     help='Hour of the day when the request is made. 7 means 07:00 to 07:59.')
    day = st.selectbox('Day of week', DAYS, index=4)
    month = st.selectbox('Month', [7, 11], index=0, format_func=lambda m: 'July' if m == 7 else 'November')
    st.divider()
    st.markdown('**Try these examples**')
    st.markdown('- City, 07:00 → high risk (morning airport rush)\n'
                '- Airport, 18:00 → low risk\n'
                '- City, 18:00 → low risk')
    st.divider()
    st.caption('Model: Random Forest (350 trees) inside a scikit-learn Pipeline. '
               'Data: Uber Request Data, Bangalore city ↔ airport, July 2016.')

probability = predict(pickup_point, hour, day, month)
is_high = probability >= THRESHOLD

# ----------------------------------------------------------------------------- header
st.title('🚕 Ride Cancellation Prediction')
st.markdown('**Case Study 66 · Machine Learning · Binary classification**  \n'
            'This app estimates the chance that a ride request will end up **Cancelled**, using only information '
            'available at the moment of booking, and explains how the model reaches that number.')

tab_predict, tab_how, tab_perf, tab_data, tab_learn = st.tabs(
    ['Prediction', 'How the model decides', 'Model performance', 'Explore the data', 'Concepts & FAQ'])

# ============================================================================= 1. PREDICTION
with tab_predict:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.metric('Estimated probability of cancellation', f'{probability:.1%}')
        st.progress(min(probability, 1.0))
        if is_high:
            st.error(f'**Higher risk**: the score is at or above the {THRESHOLD:.0%} decision threshold.')
        else:
            st.success(f'**Lower risk**: the score is below the {THRESHOLD:.0%} decision threshold.')
    with c2:
        st.metric('Average request in the data', f'{BASE_RATE:.1%}',
                  help='Share of all 6,745 requests that were cancelled.')
        st.caption('About 1 in 5 of all requests in the dataset were cancelled.')
    with c3:
        hist = data[(data['pickup_point'] == pickup_point) & (data['hour'] == hour)]
        if len(hist):
            st.metric(f'Real history: {pickup_point} at {hour:02d}:00', f"{hist['target_cancelled'].mean():.1%}",
                      help='What actually happened to past requests with the same pickup point and hour.')
            st.caption(f"{hist['target_cancelled'].sum()} of {len(hist)} past requests like this were cancelled.")

    with st.expander('Why is the model\'s score higher than the real history?'):
        st.markdown(
            "The model was trained with `class_weight='balanced'`, which makes each cancellation count about 4× more "
            'than a normal ride so the model does not ignore the rare class. A side effect is that the scores are pushed '
            'upwards: they are best read as a **risk score for ranking requests** (higher = riskier), not as an exact '
            'real-world percentage. The ranking is what ROC-AUC measures. Making the scores match real frequencies is '
            'called **probability calibration** and is listed as a future improvement.')

    st.subheader('Why this score?')
    st.markdown(
        'The model learned that **pickup point** and **hour of day** matter most. '
        'The chart shows the model\'s risk for every hour, for both pickup points. '
        'The large dot is your request; the dashed line is the 50% threshold that separates "higher" from "lower" risk.')

    curve = pd.DataFrame([{'hour': h, 'pickup_point': p, 'probability': predict(p, h, day, month)}
                          for p in ['Airport', 'City'] for h in range(24)])
    lines = alt.Chart(curve).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('hour:O', title='Request hour'),
        y=alt.Y('probability:Q', title='Predicted probability of cancellation', axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('pickup_point:N', scale=PICKUP_COLORS, title='Pickup point'),
        tooltip=[alt.Tooltip('pickup_point:N', title='Pickup'), alt.Tooltip('hour:O', title='Hour'),
                 alt.Tooltip('probability:Q', title='Risk', format='.1%')])
    rule = alt.Chart(pd.DataFrame({'y': [THRESHOLD]})).mark_rule(strokeDash=[6, 4], color=GREY).encode(y='y:Q')
    you = alt.Chart(pd.DataFrame([{'hour': hour, 'probability': probability, 'pickup_point': pickup_point}])) \
        .mark_point(size=320, filled=True, color='black', opacity=0.85).encode(x='hour:O', y='probability:Q')
    st.altair_chart((lines + rule + you).properties(height=340), width='stretch')

    other = 'Airport' if pickup_point == 'City' else 'City'
    p_other = predict(other, hour, day, month)
    peak_hour = int(curve[curve.pickup_point == pickup_point].sort_values('probability').iloc[-1]['hour'])
    st.markdown(
        f'- **Same time, other pickup point:** an {other} request at {hour:02d}:00 would score **{p_other:.1%}**.\n'
        f'- **Riskiest hour for {pickup_point} pickups:** {peak_hour:02d}:00.\n'
        '- **Day and month barely move the score.** The data covers only five weekdays of one week, '
        'so the model had almost nothing to learn from them (see "How the model decides").')

    st.subheader('How should this score be used?')
    if is_high:
        st.warning('Requests like this were often cancelled in the past. Sensible uses: tell the rider early, '
                   'offer the driver an incentive, or plan more drivers for this time and place. '
                   'Do **not** use the score to penalise a driver or rider: only about 1 in 3 "higher risk" alerts '
                   'turned out to be a real cancellation in testing.')
    else:
        st.info('Requests like this were usually not cancelled. The score is an estimate, not a guarantee: '
                'the model still misses about 18% of real cancellations.')

# ============================================================================= 2. HOW IT DECIDES
with tab_how:
    st.markdown('This tab follows **your current request** through the same steps the model uses.')

    st.subheader('Step 1 · Your request as a row of data')
    row = make_row(pickup_point, hour, day, month)
    st.dataframe(row, hide_index=True, width='stretch')
    st.caption('`is_weekend` is not asked for: it is calculated from the day (1 for Saturday or Sunday, else 0). '
               'These five columns are exactly the features the model was trained on.')

    with st.expander('Which columns were used, and which were deliberately left out?', expanded=False):
        st.markdown(
            '| Column in the raw data | Used? | Why |\n|---|---|---|\n'
            '| Pickup point | ✅ Feature | Known at booking time. City and Airport behave very differently. |\n'
            '| Request timestamp | ✅ Turned into hour, day, month, weekend | Known at booking time. |\n'
            '| Status | 🎯 Target | `Cancelled` → 1, `Trip Completed` / `No Cars Available` → 0 |\n'
            '| Driver id | ❌ Excluded | Not known yet when the customer books (data leakage). |\n'
            '| Drop timestamp | ❌ Excluded | Only exists after a finished trip, i.e. future information (data leakage). |\n'
            '| Request id | ❌ Excluded | Just a serial number, with no meaning for prediction. |')
        st.caption('Data leakage = training on information that would not be available at prediction time. '
                   'It makes a model look great in testing and fail in real use.')

    st.subheader('Step 2 · Preprocessing: turning the row into numbers')
    pre = model.named_steps['preprocess']
    transformed = pre.transform(row)
    transformed = transformed.toarray() if hasattr(transformed, 'toarray') else np.asarray(transformed)
    names = pre.get_feature_names_out()

    def explain(name):
        if name.startswith('cat__pickup_point_'):
            return 'One-hot encoding: 1 if the pickup point is ' + name.split('_')[-1] + ', else 0.'
        if name.startswith('cat__day_of_week_'):
            return ('One-hot: 1 if the day is ' + name.split('_')[-1] +
                    '. Days not seen in training (e.g. Friday) become all zeros.')
        col = name.replace('num__', '')
        return f'Standard scaling: (value − training mean) ÷ training std. Here {col} = {row.iloc[0][col]}.'

    def note(name, value):
        if name == 'num__month' and abs(value) > 3:
            return ('Far outside the training range: the mixed date formats in the raw file meant training months were '
                    'read as 11/12 (see report limitations). Month has almost no effect on the score.')
        return ''

    vec = pd.DataFrame({'Model input column': names, 'Value': np.round(transformed[0], 3),
                        'What it means': [(explain(n) + ' ' + note(n, v)).strip() for n, v in zip(names, transformed[0])]})
    st.table(vec.set_index('Model input column'))
    st.markdown('- **Categories** (pickup point, day) are first filled if missing (most frequent value), then '
                '**one-hot encoded** into 0/1 columns, because a model cannot multiply the word "City".\n'
                '- **Numbers** (hour, month, weekend) are first filled if missing (median), then **standard-scaled** '
                'so they sit on a similar scale.\n'
                '- All of this lives **inside the saved Pipeline**, and was learned only from the training data, '
                'which is why the app and the training script always process inputs identically.')

    st.subheader('Step 3 · 350 decision trees vote')
    forest = model.named_steps['model']
    tree_probs = np.array([t.predict_proba(transformed)[0, 1] for t in forest.estimators_])
    leaning = int((tree_probs >= 0.5).sum())
    t1, t2 = st.columns([1, 1.6])
    with t1:
        st.metric('Trees leaning towards "cancelled"', f'{leaning} of {len(tree_probs)}')
        st.metric('Average of all tree scores = final probability', f'{tree_probs.mean():.1%}')
        st.caption('A Random Forest trains many decision trees, each on a random sample of the training rows '
                   'and features. Each tree is a flowchart of yes/no questions like "Is pickup City?" and '
                   '"Is the hour before 10?". The forest averages their answers, which is more stable than '
                   'trusting a single tree.')
    with t2:
        hist_df = pd.DataFrame({'tree_probability': tree_probs})
        st.altair_chart(alt.Chart(hist_df).mark_bar(color=RED if is_high else TEAL).encode(
            x=alt.X('tree_probability:Q', bin=alt.Bin(maxbins=20), title='Score given by an individual tree',
                    axis=alt.Axis(format='%')),
            y=alt.Y('count():Q', title='Number of trees')).properties(height=260, title='How the 350 trees scored your request'),
            width='stretch')

    st.subheader('Step 4 · Threshold → final label')
    st.markdown(f'Final probability **{probability:.1%}** {"≥" if is_high else "<"} threshold **{THRESHOLD:.0%}** '
                f'→ label **{"Higher risk" if is_high else "Lower risk"}**. '
                'Moving the threshold trades missed cancellations against false alarms; try it in "Model performance".')

    st.subheader('Which inputs matter most?')
    if importance is not None:
        imp = importance.copy()
        imp['importance'] = imp['importance'].clip(lower=0)
        st.altair_chart(alt.Chart(imp).mark_bar(color=TEAL).encode(
            x=alt.X('importance:Q', title='Drop in ROC-AUC when this column is shuffled'),
            y=alt.Y('feature:N', sort='-x', title=None),
            tooltip=[alt.Tooltip('feature:N'), alt.Tooltip('importance:Q', format='.3f')]).properties(height=220),
            width='stretch')
        st.markdown('This is **permutation importance**: shuffle one column on the test set and measure how much worse '
                    'the model gets. **Pickup point** and **hour** carry almost all the signal. Day, month and weekend add '
                    'almost nothing because the data covers only five weekdays of one week, so those columns barely vary.')

# ============================================================================= 3. PERFORMANCE
with tab_perf:
    st.markdown('The model was trained on 75% of the data (5,058 requests) and tested on the other **25% (1,687 requests)** '
                'that it never saw during training. All numbers below come from that test set.')
    if metrics_df is not None:
        rf = metrics_df.set_index('model').loc['random_forest']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('ROC-AUC', f"{rf['roc_auc']:.3f}", help='0.5 = coin toss, 1.0 = perfect ranking.')
        m2.metric('Recall', f"{rf['recall']:.1%}", help='Share of real cancellations the model caught.')
        m3.metric('Precision', f"{rf['precision']:.1%}", help='Share of alerts that were real cancellations.')
        m4.metric('Average precision', f"{rf['average_precision']:.3f}",
                  help=f'Random guessing would score about {BASE_RATE:.3f}.')

        st.subheader('Random Forest vs Logistic Regression')
        show = metrics_df.rename(columns={'model': 'Model', 'roc_auc': 'ROC-AUC', 'average_precision': 'Avg precision',
                                          'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'accuracy': 'Accuracy'})
        show['Model'] = show['Model'].str.replace('_', ' ').str.title()
        st.dataframe(show.style.format({c: '{:.3f}' for c in show.columns if c != 'Model'}),
                     hide_index=True, width='stretch')
        st.caption('Selection rule in src/train.py: highest average precision, then ROC-AUC. Random Forest wins on 5 of 6 '
                   'metrics. Logistic Regression has slightly higher recall only because it raises more false alarms.')

    if preds is not None:
        st.subheader('Try the decision threshold yourself')
        st.markdown('The model outputs a probability. A **threshold** turns it into a yes/no alert. '
                    'The app uses 0.50. Move the slider to see the trade-off on the real test set.')
        thr = st.slider('Decision threshold', 0.05, 0.95, THRESHOLD, 0.05)
        y_true = preds['y_true']
        y_pred = (preds['random_forest'] >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Recall (cancellations caught)', f'{recall_score(y_true, y_pred, zero_division=0):.1%}')
        k2.metric('Precision (alerts that were right)', f'{precision_score(y_true, y_pred, zero_division=0):.1%}')
        k3.metric('F1 score', f'{f1_score(y_true, y_pred, zero_division=0):.3f}')
        k4.metric('Accuracy', f'{accuracy_score(y_true, y_pred):.1%}')

        cm_col, txt_col = st.columns([1, 1.2])
        with cm_col:
            cm = pd.DataFrame([
                {'Actual': 'Not cancelled', 'Predicted': 'Not cancelled', 'n': tn, 'kind': 'Correct'},
                {'Actual': 'Not cancelled', 'Predicted': 'Cancelled', 'n': fp, 'kind': 'False alarm'},
                {'Actual': 'Cancelled', 'Predicted': 'Not cancelled', 'n': fn, 'kind': 'Missed'},
                {'Actual': 'Cancelled', 'Predicted': 'Cancelled', 'n': tp, 'kind': 'Correct'}])
            base = alt.Chart(cm).encode(x=alt.X('Predicted:N', sort=['Not cancelled', 'Cancelled'],
                                                axis=alt.Axis(labelAngle=0, orient='top')),
                                        y=alt.Y('Actual:N', sort=['Not cancelled', 'Cancelled']))
            heat = base.mark_rect().encode(color=alt.Color('kind:N', legend=None, scale=alt.Scale(
                domain=['Correct', 'False alarm', 'Missed'], range=['#cfe5e2', '#f6d2cf', '#f9e3bd'])))
            text = base.mark_text(fontSize=22, fontWeight='bold').encode(text='n:Q')
            label = base.mark_text(dy=22, fontSize=11, color='#444').encode(text='kind:N')
            st.altair_chart((heat + text + label).properties(height=280, title=f'Confusion matrix at threshold {thr:.2f}'),
                            width='stretch')
        with txt_col:
            st.markdown(f'- ✅ **{tp}** cancellations correctly flagged (true positives)\n'
                        f'- ⚠️ **{fn}** cancellations missed (false negatives)\n'
                        f'- 🔔 **{fp}** false alarms (false positives)\n'
                        f'- ✅ **{tn}** normal rides correctly left alone (true negatives)')
            st.markdown('**Lower threshold** → more alerts → catch more cancellations, but more false alarms.  \n'
                        '**Higher threshold** → fewer alerts → fewer false alarms, but more missed cancellations.')
            always_no = (y_true == 0).mean()
            st.info(f'**Why not just use accuracy?** A useless model that always says "not cancelled" would score '
                    f'**{always_no:.1%} accuracy** while catching zero cancellations, because only {y_true.mean():.1%} '
                    'of test requests were cancelled. That is why recall, precision and ROC-AUC matter more here.')

        st.subheader('ROC and precision–recall curves')
        fpr, tpr, _ = roc_curve(y_true, preds['random_forest'])
        prec, rec, _ = precision_recall_curve(y_true, preds['random_forest'])
        r1, r2 = st.columns(2)
        with r1:
            roc = pd.DataFrame({'False positive rate': fpr, 'True positive rate (recall)': tpr})
            diag = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(strokeDash=[5, 4], color=GREY).encode(x='x', y='y')
            st.altair_chart((alt.Chart(roc).mark_line(color=RED, strokeWidth=2.5).encode(
                x='False positive rate:Q', y='True positive rate (recall):Q') + diag)
                .properties(height=300, title='ROC curve (dashed = random guessing)'), width='stretch')
            st.caption('The more the curve bows to the top-left, the better the model separates cancelled from '
                       'non-cancelled requests. The area under it is the ROC-AUC (0.764).')
        with r2:
            pr = pd.DataFrame({'Recall': rec, 'Precision': prec})
            base_line = alt.Chart(pd.DataFrame({'y': [BASE_RATE]})).mark_rule(strokeDash=[5, 4], color=GREY).encode(y='y:Q')
            st.altair_chart((alt.Chart(pr).mark_line(color=TEAL, strokeWidth=2.5).encode(
                x='Recall:Q', y=alt.Y('Precision:Q', scale=alt.Scale(domain=[0, 1]))) + base_line)
                .properties(height=300, title='Precision–recall curve (dashed = random guessing)'), width='stretch')
            st.caption('Shows the trade-off: catching more cancellations (moving right) lowers precision. '
                       'The area under it is the average precision (0.373), about twice the random baseline.')

# ============================================================================= 4. DATA
with tab_data:
    st.markdown('**Source:** the public *Uber Request Data* dataset (Uber Supply–Demand Gap case study, IIIT Bangalore / upGrad), '
                '[GitHub mirror](https://github.com/dittakaviram/Data-Science/blob/master/Uber%20Supply%20Demand%20Gap/Uber%20Request%20Data.csv). '
                'Ride requests between Bangalore city and the airport, 11–15 July 2016.')
    d1, d2, d3, d4 = st.columns(4)
    d1.metric('Requests', f'{len(raw):,}')
    d2.metric('Columns', raw.shape[1])
    d3.metric('Duplicate rows', int(raw.duplicated().sum()))
    d4.metric('Cancelled', f'{BASE_RATE:.1%}')

    st.subheader('What happened to each request?')
    counts = data['status'].value_counts().reindex(OUTCOME_ORDER).reset_index()
    counts.columns = ['status', 'requests']
    counts['share'] = counts['requests'] / counts['requests'].sum()
    oc1, oc2 = st.columns([1.3, 1])
    with oc1:
        st.altair_chart(alt.Chart(counts).mark_bar().encode(
            x=alt.X('status:N', sort=OUTCOME_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('requests:Q', title='Requests'),
            color=alt.Color('status:N', scale=OUTCOME_COLORS, legend=None),
            tooltip=['status', 'requests', alt.Tooltip('share:Q', format='.1%')]).properties(height=280),
            width='stretch')
    with oc2:
        st.markdown('- **Trip Completed**: the ride happened.\n'
                    '- **Cancelled**: a driver was assigned but the ride was cancelled. → **target = 1**\n'
                    '- **No Cars Available**: no driver was free at all. → target = 0\n\n'
                    'No Cars Available is kept separate from Cancelled because it is a **supply shortage** '
                    'and needs a different fix (more drivers), not a cancellation alert.')
        st.caption('Only about 1 in 5 requests is a cancellation, so the classes are imbalanced. '
                   "The models use class_weight='balanced' to stop them ignoring the rare class.")

    st.subheader('Cancellation rate by hour and pickup point')
    rate = data.groupby(['pickup_point', 'hour'])['target_cancelled'].agg(['mean', 'size']).reset_index()
    st.altair_chart(alt.Chart(rate).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('hour:O', title='Request hour'),
        y=alt.Y('mean:Q', title='Share of requests cancelled', axis=alt.Axis(format='%')),
        color=alt.Color('pickup_point:N', scale=PICKUP_COLORS, title='Pickup point'),
        tooltip=['pickup_point', 'hour', alt.Tooltip('mean:Q', title='Cancel rate', format='.1%'),
                 alt.Tooltip('size:Q', title='Requests')]).properties(height=320), width='stretch')
    st.markdown('**City pickups in the early morning** (roughly 04:00–10:00) are cancelled far more often than anything '
                'else. These are mostly people going to the airport for morning flights. Airport pickups are rarely cancelled. '
                'This pattern is what the model learned.')

    st.subheader('All outcomes by hour')
    which = st.radio('Show pickups from', ['Both', 'City', 'Airport'], horizontal=True)
    sub = data if which == 'Both' else data[data['pickup_point'] == which]
    by_hour = sub.groupby(['hour', 'status']).size().reset_index(name='requests')
    st.altair_chart(alt.Chart(by_hour).mark_bar().encode(
        x=alt.X('hour:O', title='Request hour'), y=alt.Y('requests:Q', title='Requests'),
        color=alt.Color('status:N', scale=OUTCOME_COLORS, sort=OUTCOME_ORDER, title='Outcome'),
        order=alt.Order('status:N'),
        tooltip=['hour', 'status', 'requests']).properties(height=320), width='stretch')
    st.markdown('- **Morning peak:** many cancellations, mostly City pickups.\n'
                '- **Evening peak (about 17:00–22:00):** "No Cars Available" dominates, mostly at the Airport, '
                'when many flights land and there are not enough cabs.')

    st.subheader('Missing values, and why they are expected')
    miss = raw.isna().sum().reset_index()
    miss.columns = ['Column', 'Missing values']
    reasons = {'Driver id': 'Blank for every "No Cars Available" request: no car, no driver.',
               'Drop timestamp': 'Blank whenever the trip did not happen (No Cars + Cancelled).'}
    miss['Reason'] = miss['Column'].map(reasons).fillna('Complete')
    st.table(miss.set_index('Column'))
    st.caption('These gaps are caused by the outcome itself, which is another reason Driver id and Drop timestamp '
               'cannot be used as features.')

    with st.expander('See the raw data'):
        st.dataframe(raw.head(200), hide_index=True, width='stretch')
        st.caption('First 200 of 6,745 rows. Note that the file mixes two date formats, '
                   'e.g. "11/7/2016 11:51" and "13-07-2016 08:18:13". The charts on this tab read both.')

# ============================================================================= 5. LEARN
with tab_learn:
    st.subheader('The project in one picture')
    st.markdown(
        '1. **Load** the raw CSV (6,745 requests, 6 columns)\n'
        '2. **Clean** column names and parse the request timestamp\n'
        '3. **Create the target**: `Cancelled` → 1, everything else → 0\n'
        '4. **Engineer features** known at booking time: pickup point, hour, day of week, month, weekend\n'
        '5. **Split** 75% train / 25% test, stratified so both keep the 18.7% cancellation rate\n'
        '6. **Preprocess** inside a Pipeline: fill blanks, one-hot encode categories, scale numbers\n'
        '7. **Train** Logistic Regression and Random Forest with balanced class weights\n'
        '8. **Evaluate** on the test set with ROC-AUC, average precision, recall and precision; pick the best\n'
        '9. **Save** the winning pipeline with joblib and serve it in this Streamlit app')

    st.subheader('Key concepts')
    concepts = {
        'Binary classification': 'Predicting one of two categories. Here: cancelled (1) or not cancelled (0).',
        'Feature / target': 'Features are the inputs (pickup point, hour, ...). The target is what we predict (target_cancelled).',
        'Data leakage': 'Using information during training that would not exist at prediction time, such as Driver id or Drop timestamp. '
                        'It gives unrealistically good test scores.',
        'Train/test split': 'The model learns from one part of the data and is graded on a separate part it never saw, '
                            'to estimate how it will do on future requests.',
        'Stratified split': 'Keeps the same share of cancellations in the training and test sets.',
        'One-hot encoding': 'Turns a category into 0/1 columns (pickup_point_City, pickup_point_Airport) so the model can use it '
                            'without inventing a fake order.',
        'Standard scaling': 'Rescales numbers to mean 0 and standard deviation 1: z = (x − mean) ÷ std.',
        'Class imbalance': "Only 18.7% of requests are cancellations. class_weight='balanced' makes mistakes on the rare class "
                           'cost about 4× more during training.',
        'Logistic Regression': 'Weighted sum of the features passed through a sigmoid curve to produce a probability. Simple and '
                               'easy to explain; used as the baseline.',
        'Random Forest': 'Many decision trees, each trained on a random sample of rows and features, whose answers are averaged. '
                         'Captures patterns like "City AND early morning" that a linear model handles poorly.',
        'Precision': 'Of the requests flagged as risky, how many were really cancelled.',
        'Recall': 'Of the requests that were really cancelled, how many were flagged.',
        'ROC-AUC': 'Probability that the model ranks a random cancelled request above a random non-cancelled one.',
        'Average precision': 'Area under the precision–recall curve; more informative than ROC-AUC when positives are rare.',
    }
    for k, v in concepts.items():
        with st.expander(k):
            st.write(v)

    st.subheader('Frequently asked questions')
    faq = {
        'Why are Driver id and Drop timestamp not used?':
            'Both are only known after the booking. Driver id is blank whenever no car was available, and Drop timestamp '
            'exists only for completed trips. Using them would leak the answer into the model.',
        'Why is "No Cars Available" not counted as a cancellation?':
            'It is a different problem: there was no driver at all (a supply gap), so the fix is more drivers, '
            'not a cancellation alert. Merging the two would hide that difference.',
        'Why does changing the day or month hardly change the score?':
            'The data covers only Monday to Friday of a single week in July 2016, so day, month and weekend barely '
            'vary and the model learned almost nothing from them. Pickup point and hour do the work.',
        'Is a 34% precision good enough?':
            'For an early-warning tool, yes: the model catches about 82% of cancellations, and a false alarm is cheap '
            '(an extra message or incentive). It is not good enough to punish anyone, which is why the app warns against that.',
        'What are the main limitations?':
            'One week of data from one city; no weather, traffic, fare, driver distance or customer history; correlation '
            'is not causation; a random rather than time-based test split; and a fixed 0.50 threshold.',
        'How could the model be improved?':
            'More data across months and weekends, extra features (wait time, driver availability, traffic, weather), '
            'hyperparameter tuning, gradient boosting models, probability calibration and a time-based validation split.',
    }
    for q, a in faq.items():
        with st.expander(q):
            st.write(a)

    st.divider()
    st.subheader('Run it yourself')
    st.code('pip install -r requirements.txt\n'
            'python src/train.py               # optional: retrain the model and regenerate figures\n'
            'python src/export_app_assets.py   # optional: refresh the files this app explains\n'
            'streamlit run app/streamlit_app.py', language='bash')
