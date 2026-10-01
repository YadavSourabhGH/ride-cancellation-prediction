# Viva Evidence and Speaking Notes

## 60-second project explanation

I built a binary classifier for the Uber Request Data case study. The model predicts whether a request is labelled `Cancelled` using only information available when the request is created: pickup point, hour, day, month, and weekend status. I kept `No Cars Available` separate because it is a supply outcome, not the same label. A random forest was selected over logistic regression after a stratified hold-out comparison. It achieved ROC-AUC 0.764, average precision 0.373, recall 0.820, and precision 0.338 on 1,687 test requests. The Streamlit app lets a user enter request-time information and see the estimated probability.

## Likely questions

### Why is this a classification problem?
The output is a class label, cancelled or not cancelled. The probability is useful for ranking requests, but the underlying task is binary classification.

### Why did you exclude Driver id and Drop timestamp?
Driver ID is often missing for unassigned requests and is not known at the initial prediction point. Drop timestamp is only known after the trip finishes. Using either would leak future information.

### Why did you not merge No Cars Available with Cancelled?
The two outcomes need different actions. A cancellation may relate to a booked request, while no-car availability indicates a supply gap. Merging them would answer a broader “request failure” question and hide that distinction.

### Why report average precision and recall?
Only 18.7% of records are cancellations. Accuracy alone can look acceptable while missing the minority class. Average precision summarises ranking quality for the positive class; recall shows how many cancellations were found.

### What does precision 0.338 mean?
At the 0.50 threshold, roughly one in three alerts was an actual cancellation in the hold-out set. The model finds many cancellations, but it also raises false alerts.

### What would you improve?
Use chronological validation, collect wait time and driver availability, calibrate probabilities, test threshold-specific costs, and monitor performance after deployment.

### What is the main limitation?
The data covers only two periods in 2016 and lacks operational variables such as weather, traffic, fare, and available supply. The result is a teaching prototype rather than a production-ready service.

## Demo sequence

1. Run `streamlit run app/streamlit_app.py`.
2. Select Airport, 18:00, Friday, July.
3. Click **Estimate cancellation risk**.
4. Explain that the score is an estimate and that the app does not use post-request fields.
5. Open the report figures and point to the separate `No Cars Available` outcome.
