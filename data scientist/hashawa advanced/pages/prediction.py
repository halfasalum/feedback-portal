import joblib
import numpy as np
import pandas as pd
import streamlit as st 
from datetime import datetime
import locale

# Set the locale to the user's default setting (works on Unix-based systems)
locale.setlocale(locale.LC_ALL, '')

# Format a number as currency

#formatted_loan = locale.currency(loan, grouping=True)

with st.container():
    st.text("Customer Loan Approval Recommendation")
    with st.form(key='loanAmountForm'):
        gender = st.selectbox(
            'Please choose customer gender',
            ("Male","Female"),
            index=None
            )
        dob = st.date_input(
            "Customer Birthdate",
            format="YYYY-MM-DD",
            value=None
        )
        marital = st.selectbox(
            'Please choose customer marital status',
            ("Married","Single"),
            index=None
            )
        income = st.number_input(
            "Customer income",
            min_value = 10000,
            value = None,
            step = None
        )
        loan = st.number_input(
            "Customer Loan Requested",
            min_value = 50000,
            value = None,
            step = None
        )
        submit = st.form_submit_button(label='Process')
    
    if(submit):
        predictionModel = joblib.load("loan_amount_prediction_model.joblib")
        if(gender == 'Male'):
            gender = 1
        else:
            gender = 0

        if(marital == 'Married'):
            marital = 1
        else:
            marital = 0
        dob = pd.to_datetime(dob)
        today = datetime.today()
        age = (today.year - dob.year)
        ratio = loan/income
        customer_data = np.array([[gender,age,income,marital]]) 
        predict_loan = predictionModel.predict(customer_data).astype(int)
        #predict_loan = locale.currency(predict_loan, grouping=True)
        st.write(f"Proposed Loan  For Customer : {predict_loan}")

        
