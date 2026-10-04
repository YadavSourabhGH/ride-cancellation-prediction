from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'ride_cancellation_model.joblib'
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title='Ride Cancellation Prediction', page_icon='🚕', layout='centered')
st.title('Ride Cancellation Prediction')
st.caption('Case Study 66 | prediction at the moment a ride request is created')
st.write('This prototype estimates the chance that a request will later be marked **Cancelled** in the source data. It uses only request-time fields, so driver and drop-off information are deliberately excluded.')

with st.form('prediction_form'):
    pickup_point = st.selectbox('Pickup point', ['Airport', 'City'])
    hour = st.slider('Request hour', min_value=0, max_value=23, value=18, help='Use local time as recorded in the dataset.')
    day = st.selectbox('Day of week', ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], index=4)
    month = st.selectbox('Month', [7, 11], index=0, format_func=lambda x: 'July' if x == 7 else 'November')
    submitted = st.form_submit_button('Estimate cancellation risk')

if submitted:
    row = pd.DataFrame([{
        'pickup_point': pickup_point,
        'hour': hour,
        'day_of_week': day,
        'month': month,
        'is_weekend': int(day in ['Saturday', 'Sunday'])
    }])
    probability = float(model.predict_proba(row)[0, 1])
    label = 'Higher risk' if probability >= 0.50 else 'Lower risk'
    st.subheader(label)
    st.metric('Estimated probability of cancellation', f'{probability:.1%}')
    if probability >= 0.50:
        st.warning('This request looks similar to records with a higher cancellation rate. A risk score can support staffing or rider messaging; it should not be used as a reason to penalize a driver or rider.')
    else:
        st.info('The request is below the model threshold used in evaluation. The score is still an estimate, not a guarantee.')
    st.caption('Model: Random Forest, selected from a logistic-regression comparison on a stratified hold-out set. See the report for metrics and limitations.')

st.divider()
st.subheader('How to run')
st.code('pip install -r requirements.txt\nstreamlit run app/streamlit_app.py', language='bash')
st.caption('Source data: Uber Request Data, public GitHub mirror linked in the report. The dataset is historical and limited to two months in 2016.')
