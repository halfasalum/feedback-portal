import streamlit as st
import pandas as pd
import numpy as np

loans = pd.read_csv("loans.csv")
loan_by_age = pd.read_csv("df_mean_loan_by_age_range.csv")
loan_by_male = pd.read_csv("df_number_loan_per_male_per_year.csv")
loan_by_female = pd.read_csv("df_number_loan_per_female_per_year.csv")


st.title("Analysis Page")


with st.container():
    st.header("Loan Application Analysis")
    st.text("Basic Analysis of Loan Data")
    st.divider()
    st.text("Average Loan Disbursement per Age Range")
    st.table(loan_by_age)
    column_1, column_2 = st.columns(2, gap='small', vertical_alignment='top')
    with column_1:
        st.text("Loan Disbursment per Year by Male gender")
        st.table(loan_by_male)
    with column_2:
        st.text("Loan Disbursment per Year by Female gender")
        st.table(loan_by_female)

st.divider()
st.text("Average Loan Disbursement per Age Group")

chart_data = pd.DataFrame(
   {
       "col1": loan_by_age['age_range'],
       "col2": loan_by_age['loan_amount'],
   }
)

st.line_chart(chart_data, x="col1", y="col2", x_label='Age Range', y_label='Loan Amount')

st.text("Loan Disbursement Per Male Per Year")

# Combine the data for plotting
chart_loan_by_gender_data = pd.DataFrame({
    "Year": loan_by_male['year'],
    "Male Loans": loan_by_male['count'],
    "Female Loans": loan_by_female['count']
})

# Melt the DataFrame for better plotting
chart_data_melted = chart_loan_by_gender_data.melt('Year', var_name='Gender', value_name='Number of Loans')

# Plot the chart
st.line_chart(chart_data_melted.pivot(index='Year', columns='Gender', values='Number of Loans'))