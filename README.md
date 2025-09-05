# Loan Approval Prediction & Fairness Dashboard  

This project is a machine learning pipeline and interactive app for predicting **loan approvals** and analyzing **fairness** and **explainability** of the model.  

It uses:  
-  **EDA** to understand the dataset  
-  **Random Forest model** for predictions  
-  **Fairness metrics** for sensitive groups  
-  **SHAP explainability** for feature importance  
-  **Streamlit dashboard** to interact with everything  

---

##  Features  

### Loan Approval Prediction  
- Input applicant details manually in the app  
- See predicted probability of loan approval  
- Adjust decision **threshold** to test different cutoffs  

### Global SHAP  
- Bar plot & beeswarm for most important features  
- SHAP summary results from training  

### Fairness Analysis  
- Evaluate fairness across sensitive features (education, employment, etc.)  
- Metrics include demographic parity & equalized odds  
