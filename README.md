# 🚕 Ride Cancellation Risk Prediction — ML Case Study 66

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ride-cancellation-prediction.streamlit.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-YadavSourabhGH%2Fride--cancellation--prediction-181717.svg?logo=github)](https://github.com/YadavSourabhGH/ride-cancellation-prediction)

> 🌐 **Live Web Application**: [https://ride-cancellation-prediction.streamlit.app/](https://ride-cancellation-prediction.streamlit.app/)  
> 📦 **GitHub Repository**: [https://github.com/YadavSourabhGH/ride-cancellation-prediction](https://github.com/YadavSourabhGH/ride-cancellation-prediction)

An end-to-end Machine Learning case study that estimates ride cancellation risk at the exact moment a booking request is initiated, using strictly pre-assignment request-time features. This system includes an end-to-end data pipeline, feature engineering, exploratory data analysis (EDA), comparative model training (Random Forest vs. Logistic Regression), evaluation metrics, and an interactive **Streamlit** web application.

---

## 📌 Table of Contents
- [Problem Overview & Objective](#-problem-overview--objective)
- [Dataset Information & Sources](#-dataset-information--sources)
- [Project Architecture & Directory Layout](#-project-architecture--directory-layout)
- [Step-by-Step Implementation](#-step-by-step-implementation)
  - [1. Data Preprocessing & Leakage Prevention](#1-data-preprocessing--leakage-prevention)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training & Comparison](#4-model-training--comparison)
  - [5. Interactive Streamlit Web Application](#5-interactive-streamlit-web-application)
- [Model Evaluation & Findings](#-model-evaluation--findings)
- [How to Run Locally (Step-by-Step)](#-how-to-run-locally-step-by-step)
- [Key Limitations & Viva Defense Notes](#-key-limitations--viva-defense-notes)

---

## 🎯 Problem Overview & Objective

When customers request on-demand rides between city centers and transit hubs (airports), high cancellation rates degrade user experience, decrease driver earnings, and cause operational deadheading.

**The ML Question:**
> *Can we estimate the probability that a ride request will later result in a **driver cancellation** using only signals observable at request time?*

### Critical Design Decision: No Data Leakage
To ensure real-world validity, features such as `Driver id` and `Drop timestamp` are strictly excluded from the feature matrix because:
1. `Driver id` is unknown at the request moment (and missing when no car is assigned).
2. `Drop timestamp` occurs in the future and only exists for completed rides.
3. `No Cars Available` is tracked as an operational supply-gap outcome and kept distinct from driver cancellations.

---

## 📊 Dataset Information & Sources

The underlying dataset is the well-known **Uber Supply-Demand Gap** case study dataset (originally curated for IIIT Bangalore / upGrad executive programs).

- **Official Public Repository / Source Mirror:** [Uber Supply Demand Gap / Uber Request Data.csv on GitHub](https://github.com/dittakaviram/Data-Science/blob/master/Uber%20Supply%20Demand%20Gap/Uber%20Request%20Data.csv)
- **Direct Raw CSV:** [raw.githubusercontent.com - Uber Request Data.csv](https://raw.githubusercontent.com/dittakaviram/Data-Science/master/Uber%20Supply%20Demand%20Gap/Uber%20Request%20Data.csv)
- **Kaggle Mirror:** [Uber Request Data on Kaggle](https://www.kaggle.com/datasets/amankumark/uber-request-data)
- **Local Raw File:** [`data/uber_request_data_raw.csv`](data/uber_request_data_raw.csv)
- **Cleaned Prepared File:** [`data/uber_request_data_prepared.csv`](data/uber_request_data_prepared.csv)

### Data Summary:
- **Total Requests:** 6,745
- **Features in Raw Data:** `Request id`, `Pickup point`, `Driver id`, `Status`, `Request timestamp`, `Drop timestamp`
- **Recorded Outcomes (`Status`):**
  - `Trip Completed`: 2,831 (42.0%)
  - `No Cars Available`: 2,650 (39.3%) — unfulfilled supply shortage
  - `Cancelled`: 1,264 (18.7%) — binary positive class (`target_cancelled = 1`)

---

## 📂 Project Architecture & Directory Layout

```text
ride_cancellation/
├── README.md                           # Comprehensive documentation & setup instructions
├── SUBMISSION_CHECKLIST.md             # Coursework verification checklist
├── requirements.txt                    # Python runtime dependencies
├── presentation.pdf                    # Slide presentation deck
├── app/
│   └── streamlit_app.py                # Interactive web app UI for real-time risk scoring
├── data/
│   ├── uber_request_data_raw.csv       # Original dataset (6,745 rows)
│   └── uber_request_data_prepared.csv  # Preprocessed & engineered dataset
├── figures/                            # Exported charts & evaluation plots
│   ├── cancel_rate_by_hour.png         # Hour-of-day cancellation rate curve
│   ├── outcome_counts.png              # Class distribution of outcomes
│   ├── outcomes_by_hour.png            # Trip completion vs cancellation vs no cars
│   ├── confusion_matrix.png            # Test set confusion matrix
│   └── evaluation_curves.png           # ROC & Precision-Recall curves
├── models/
│   └── ride_cancellation_model.joblib # Serialized scikit-learn pipeline (Preprocessor + RF)
├── report/
│   ├── final_report.md                 # Detailed academic report
│   ├── final_report.pdf                # Compiled final report document
│   ├── model_metrics.csv               # Comparison metrics table
│   ├── classification_report.txt       # Per-class precision, recall, f1
│   └── analysis_summary.json           # Programmatic summary metrics
├── evidence/
│   └── viva_notes.md                   # Viva questions, answers, and demo script
└── src/
    └── train.py                        # Complete pipeline script: ETL, EDA, Train & Eval
```

---

## 🛠️ Step-by-Step Implementation

### 1. Data Preprocessing & Leakage Prevention
In [`src/train.py`](src/train.py):
- Normalized column naming conventions to snake_case.
- Parsed datetime strings safely into timestamps.
- Formulated the binary classification target:
  $$\text{target\_cancelled} = \begin{cases} 1 & \text{if } \text{status} = \text{'Cancelled'} \\ 0 & \text{otherwise} \end{cases}$$
- Isolated `No Cars Available` from the cancellation class so the classifier targets active driver rejections rather than platform vehicle shortages.

### 2. Exploratory Data Analysis (EDA)
EDA scripts automatically generate high-resolution figures in `figures/`:
1. **Outcome Breakdown:** Visualizes the 42.0% completion, 39.3% car unavailability, and 18.7% cancellation distributions.
2. **Temporal & Route Patterns:**
   - **Morning Peak (05:00 - 09:00):** Massive spike in cancellations for **City $\rightarrow$ Airport** requests (drivers cancel to avoid deadheading back empty).
   - **Evening Peak (17:00 - 22:00):** Dominated by **No Cars Available** at the Airport due to extreme incoming flight demand.

### 3. Feature Engineering
Only attributes available at request creation are transformed:
- `pickup_point`: Categorical (`'Airport'`, `'City'`)
- `hour`: Discrete integer (0 to 23)
- `day_of_week`: Categorical (`'Monday'`, ..., `'Sunday'`)
- `month`: Numerical (7, 11)
- `is_weekend`: Binary flag derived from day of the week
- `time_band`: Categorical bins (`'Night'`, `'Morning'`, `'Afternoon'`, `'Evening'`, `'Late evening'`)

### 4. Model Training & Comparison
The pipeline uses `scikit-learn` `ColumnTransformer` with `Pipeline`:
- **Categorical Features:** Imputed with mode $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
- **Numerical Features:** Imputed with median $\rightarrow$ `StandardScaler()`.
- **Class Imbalance Handling:** Addressed via `class_weight='balanced'`.
- **Train/Test Split:** Stratified 75% train (5,058 rows) / 25% test (1,687 rows), `random_state=42`.

#### Models Compared:
1. **Logistic Regression:** Linear baseline with balanced weights.
2. **Random Forest Classifier:** Ensemble (350 estimators, `min_samples_leaf=8`, balanced weights).

### 5. Interactive Streamlit Web Application
- **Live Deployment:** [https://ride-cancellation-prediction.streamlit.app/](https://ride-cancellation-prediction.streamlit.app/)
- **Code:** [`app/streamlit_app.py`](app/streamlit_app.py)
- **Features:**
  - Interactive UI allowing operations managers or users to choose pickup location, request hour, day, and month.
  - Computes real-time cancellation probability using `ride_cancellation_model.joblib`.
  - Flags requests as **Higher risk** ($\ge 50\%$) or **Lower risk** ($< 50\%$) with contextual advisory messaging.

---

## 📈 Model Evaluation & Findings

Evaluated on the stratified hold-out test set ($N = 1,687$):

| Metric | Random Forest (Selected) | Logistic Regression |
|---|:---:|:---:|
| **ROC-AUC** | **0.7639** | 0.7520 |
| **Average Precision (PR-AUC)** | **0.3731** | 0.3533 |
| **Recall (Cancellations)** | **81.96%** | **85.76%** |
| **Precision** | **33.77%** | 31.26% |
| **F1-Score** | **0.4783** | 0.4582 |
| **Overall Accuracy** | **66.51%** | 62.00% |

### Key Takeaway:
The Random Forest model captures **82% of all actual cancellations** (high recall). The trade-off is a precision of ~33.8%, which is expected given the base cancellation rate of 18.7%. This makes it well-suited as an operational early-warning system (e.g., dispatching priority alerts or rider notifications) rather than a punitive automated blocker.

---

## 🚀 How to Run Locally (Step-by-Step)

Follow these exact steps to clone, set up, train, and launch the application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/YadavSourabhGH/ride-cancellation-prediction.git
cd ride-cancellation-prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train Model & Re-generate Figures
To re-run the entire ETL, generate new plots, and save the serialized model pipeline:
```bash
python src/train.py
```
*Outputs will be refreshed in `figures/`, `models/`, and `report/`.*

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser at **`http://localhost:8501`** (or the port displayed in your terminal).

---

## 🔍 Key Limitations & Viva Defense Notes

1. **Observational & Temporal Scope:** The dataset covers five days in July 2016. Because of historical mixed date formatting (`DD/MM/YYYY` vs `DD-MM-YYYY`), common American date parsers identify records under July and November. This is documented as a primary data limitation in the final report.
2. **Missing Operational Context:** The dataset does not contain real-time driver distance, traffic congestion indices, fare pricing / surge multipliers, or local weather conditions.
3. **Threshold Calibration:** The default decision threshold is 0.50. In production, this threshold should be adjusted based on the specific cost asymmetry between false alerts and missed cancellations.

---

## 📜 License & Citation
This project is open-sourced under the MIT License for educational and research purposes. Dataset credit goes to the IIIT-Bangalore / upGrad curriculum authors.
