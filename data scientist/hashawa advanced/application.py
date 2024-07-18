import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime

loans = pd.read_csv("loans.csv")
loan_by_age = pd.read_csv("df_mean_loan_by_age_range.csv")
loan_by_male = pd.read_csv("df_number_loan_per_male_per_year.csv")
loan_by_female = pd.read_csv("df_number_loan_per_female_per_year.csv")




st.set_page_config(layout="wide")

st.header("Loan Analysis and Prediction System")

home, recomendation, loanPrediction, customer, graph, metrics = st.tabs(['Home','Loan Recomendation','Loan Prediction','Customer Analaysis','Dataset Graphs','Metrics'])

with home:
    st.subheader("Introduction")
    st.text('This project aims to develop a comprehensive loan analysis and prediction system for a credit company. Baseline [Tanzania, Dar es salaam]')
    st.text('The system leverages machine learning techniques to analyze customer datasets, provide loan approval recommendations, predict loan amounts, and categorize customers based on their behaviors.')
    st.text('The project focuses on real-time data analysis using a streaming approach in Python to ensure timely and accurate decision-making.')
    st.subheader("Objectives")
    st.text("Dataset Analysis: Analyze customer information, loan history, and payment history to identify patterns and insights.")
    st.text("Loan Approval Recommendation: Develop a model to recommend whether a loan should be approved based on key customer attributes.")
    st.text("Loan Amount Prediction: Predict the optimal loan amount for approved customers using predictive modeling techniques.")
    st.text("Customer Categorization: Categorize customers into different risk categories based on their credit profile and payment behavior.")
    st.subheader("Conclusion")
    st.text("This project will enable the credit company to make data-driven decisions, improve loan approval accuracy, optimize loan amounts, and better understand customer behavior.")
    st.text("The real-time processing capability will ensure timely responses to customer applications and enhance overall operational efficiency.")


with recomendation:
    st.subheader('Loan Approval Recomendation')
    st.text("This section will provide the recommndation for loan approval based on the inpt provided")
    with st.form(key='loanApproval'):
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
            recomendationModel = joblib.load("loan_default_model.joblib")
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
            customer_data = np.array([[gender,age,marital,income,loan,ratio]])
            recomend = recomendationModel.predict(customer_data).astype(int)
            if (recomend == 5):
                st.write("Recommended For Approval")
                predictionModel = joblib.load("loan_amount_prediction_model.joblib")
                customer_data = np.array([[gender,age,income,marital]]) 
                predict_loan = predictionModel.predict(customer_data).astype(int)
                st.write(f"Proposed Loan  For Customer Maximum To : {predict_loan}")

            else:
                st.write("NOT Recommended For Approval")


with loanPrediction:
    st.subheader('Loan Amount Recomendation')
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
        st.write(f"Proposed Loan  For Customer To Maximum Of : {predict_loan}")

with customer:
    st.text('Customer Analaysis')

with graph:
    st.text('Dataset Graphs')

with metrics:
    st.text('Metrics')


