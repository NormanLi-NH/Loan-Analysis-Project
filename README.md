# Loan Analysis and Visualization

## Overview

This project provides tools for analyzing loan data, visualizing trends, and predicting interest rates using Python. It includes various functions for:

- **Loan Amount to Annual Income Ratio**: Calculate and analyze the ratio of loan amounts to annual income.
- **Annual Installment to Annual Income**: Assess the proportion of annual installments relative to annual income.
- **Total Interest Analysis**: Analyze the total interest paid over the loan term.
- **Loan Purpose Count**: Visualize the distribution of loan purposes.
- **Count by State**: Display the number of loans issued by state.
- **Mean Interest Rate by Grade**: Show average interest rates categorized by loan grade.
- **Mean Interest Rate by Home Ownership**: Analyze average interest rates based on home ownership status.
- **Interest Rate Distribution by Loan Term**: Visualize interest rates based on the term of the loan.
- **Interest Rates Over Time**: Plot mean interest rates over time alongside FED interest rates.
- **Normalized Interest Rates Over Time**: Normalize and compare loan and FED interest rates over time.

The project also includes a user-friendly GUI for easy interaction and data analysis.

## Requirements

To run this project, you need the following:

- **Python 3.x**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Tkinter** (included with standard Python installations)

You can install the required libraries using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn

Data
This project requires a dataset containing loan information. Make sure your dataset includes the following columns:

annual_inc: Annual income of the borrower.
avg_cur_bal: Average current balance.
grade: Loan grade (A-G).
state: Borrower’s state.
purpose: Purpose of the loan (e.g., home improvement, debt consolidation).
int_rate: Interest rate of the loan.
issue_date: Date the loan was issued.

Usage
To get started, follow these steps:

Clone the repository:
bash


git clone <repository-url>
cd loan-analysis
Load your dataset into the inner_join_df DataFrame.
Run the script:
bash


python loan_analysis.py
Use the GUI to:
Input borrower information for interest rate calculations.
Select various plots to visualize loan data.

Functions
Data Visualization Functions
plot_loan_purpose(): Visualizes the distribution of loan purposes using a pie chart.
plot_state_counts(): Displays a bar chart of loan counts by state.
plot_grade_analysis(): Shows the mean interest rates categorized by loan grade.
plot_home_ownership_analysis(): Analyzes mean interest rates based on home ownership status.
plot_term_analysis(): Visualizes the distribution of interest rates by loan term.
plot_interest_rate_over_time(): Plots mean interest rates over time alongside FED interest rates.
plot_normalised_interest_rate_over_time(): Normalizes and compares loan and FED interest rates over time.

Machine Learning
This project includes a linear regression model that predicts interest rates based on annual_inc, avg_cur_bal, and grade_score.

GUI Functions
A Tkinter-based GUI allows users to input their financial data and generate visualizations easily.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any changes or improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Pandas for data manipulation.
Matplotlib and Seaborn for data visualization.
Scikit-learn for machine learning functionalities.
Tkinter for creating the GUI.
