import pyodbc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox

# Load data from CSV files
df_customer = pd.read_csv("customer.csv")
df_state_region = pd.read_csv("state_region.csv")
df_loan = pd.read_csv("loan.csv")
df_fed = pd.read_csv("FEDFUNDS.csv")

print(df_customer.head())
print(df_loan.info())

##Inner Join Customer and Loan Tables
inner_join_df = pd.merge(df_customer, df_loan, on='customer_id', how='outer')

######################################################################################

###Loan Amount to Annual Income

    
    ##Remove the outliers for annual income 
def plot_loan_income_ratio():
    lower_percentile = 0.01  # 1st percentile
    upper_percentile = 0.99  # 99th percentile

    # Calculate bounds
    lower_bound = inner_join_df['annual_inc'].quantile(lower_percentile)
    upper_bound = inner_join_df['annual_inc'].quantile(upper_percentile)

    # Remove outliers for annual income
    Q1 = inner_join_df['annual_inc'].quantile(0.25)
    Q3 = inner_join_df['annual_inc'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    inner_join_df = inner_join_df[(inner_join_df['annual_inc'] >= lower_bound) & (inner_join_df['annual_inc'] <= upper_bound)]

    # Loan Amount to Annual Income Ratio (as percentage)
    inner_join_df["loan_inc_ratio"] = (inner_join_df["loan_amount"] / inner_join_df["annual_inc"]) * 100

    # Drop NaN values for the plot
    plot_df = inner_join_df.dropna(subset=['loan_inc_ratio'])

    # Define bins and labels
    bins = [0, 5, 10, 15, 20, 25, 50, 100, 200]  # Adjusted for percentage
    labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-50%', '50-100%', '100%+']

    # Categorize the loan_inc_ratio into bins
    plot_df['ratio_bins'] = pd.cut(plot_df['loan_inc_ratio'], bins=bins, labels=labels, right=False)
    count_by_bins = plot_df['ratio_bins'].value_counts().sort_index()

    # Plotting the bar chart
    plt.figure(figsize=(12, 8))
    count_by_bins.plot(kind='bar', color='skyblue', width=0.5)  # Set bar width
    plt.title('Loan Amount to Annual Income Ratio (%)')
    plt.xlabel('Loan to Income Ratio Ranges')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

######################################################################################


    ###Installment Analysis
def plot_installment_analysis():
    # Calculate the ratio and convert to percentage
    inner_join_df['installment_ratio'] = (inner_join_df['installment'] * 12 / inner_join_df['annual_inc']) * 100

    # Remove outliers using the IQR method
    Q1 = inner_join_df['installment_ratio'].quantile(0.25)
    Q3 = inner_join_df['installment_ratio'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    filtered_df = inner_join_df[(inner_join_df['installment_ratio'] >= lower_bound) & (inner_join_df['installment_ratio'] <= upper_bound)]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.hist(filtered_df['installment_ratio'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Annual Installment to Annual Income')
    plt.xlabel('Percentage of Installment to Income (%)')
    plt.ylabel('Number of Customers')
    plt.grid(axis='y')

    # Adding the mean line
    mean_percentage = filtered_df['installment_ratio'].mean()
    plt.axvline(mean_percentage, color='red', linestyle='dashed', linewidth=1, label='Mean Percentage')
    plt.legend()
    plt.tight_layout()
    plt.show()



######################################################################################

###Total Interest Analysis
def plot_total_interest():
    # Convert 'term' string to integer
    inner_join_df['term'] = inner_join_df['term'].str.replace(' months', '').astype(int)


    # Calculate total interest
    inner_join_df['Total_Interest'] = (inner_join_df['installment'] * inner_join_df['term']) - inner_join_df['loan_amount']

    # Drop NaN values for the plot
    plot_df = inner_join_df.dropna(subset=['Total_Interest'])

    # Remove negative values and calculate Q1 and Q3
    filtered_df = plot_df[plot_df['Total_Interest'] >= 0]

    Q1 = filtered_df['Total_Interest'].quantile(0.25)
    Q3 = filtered_df['Total_Interest'].quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Further filter the DataFrame to remove outliers
    filtered_df = filtered_df[(filtered_df['Total_Interest'] >= lower_bound) & (filtered_df['Total_Interest'] <= upper_bound)]

    # Calculate mean and median of the filtered data
    mean_value = filtered_df['Total_Interest'].mean()
    median_value = filtered_df['Total_Interest'].median()

    # Visualizations
    plt.figure(figsize=(12, 6))
    plt.boxplot(filtered_df['Total_Interest'], vert=False)
    plt.title('Total Interest on Loans')
    plt.xlabel('Total Interest ($)')
    plt.grid(axis='x')
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label='Mean Total Interest')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label='Median Total Interest')
    plt.legend()
    plt.show()

######################################################################################

def plot_loan_purpose():
    # Get value counts for the 'purpose' column
    purpose_counts = inner_join_df["purpose"].value_counts()

    # Create a DataFrame for plotting
    purpose_df = purpose_counts.reset_index()
    purpose_df.columns = ['purpose', 'count']

    # Plotting the pie chart
    plt.figure(figsize=(6, 6))
    wedges, texts, autotexts = plt.pie(
        purpose_df['count'], 
        labels=None,  
        autopct='%1.1f%%', 
        startangle=140, 
        explode=[0.1] * len(purpose_df)  # Slightly explode each slice for clarity
    )

    # Adding a legend
    plt.legend(wedges, purpose_df['purpose'], title="Loan Purposes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Adjust text properties
    for text in autotexts:
        text.set_color('white')  # Change color of the percentage text for visibility
        text.set_fontsize(10)    # Adjust font size for better fitting

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Loan Purpose Count', fontsize=16)

    # Show plot
    plt.tight_layout()
    plt.show()

######################################################################################

def plot_state_counts():
    # Count occurrences of each state
    state_counts = inner_join_df['state'].value_counts()

    # Plotting the data
    plt.figure(figsize=(12, 6))
    state_counts.plot(kind='bar', color='skyblue')

    # Adding titles and labels
    plt.title('Count of Values by State', fontsize=16)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # Rotate x labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()


######################################################################################
def plot_grade_analysis():
    ###Grade analysis
    mean_grade_rate = inner_join_df.groupby('grade')['int_rate'].mean().reset_index()

    #plotting
    plt.figure(figsize=(10, 6))
    plt.bar(mean_grade_rate['grade'], mean_grade_rate['int_rate'], color='skyblue')
    plt.title('Mean Interest Rate by Grade')
    plt.xlabel('Grade')
    plt.ylabel('Mean Interest Rate')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()



######################################################################################
def plot_home_owership_analysis():
    ##home_ownership analysis
    mean_own_rate = inner_join_df.groupby("home_ownership")['int_rate'].mean().reset_index()
    # plotting
    plt.figure(figsize=(10, 6))
    plt.bar(mean_own_rate['home_ownership'], mean_own_rate['int_rate'], color='skyblue')
    plt.title('Mean Interest Rate by Home Ownership')
    plt.xlabel('Home Ownership')
    plt.ylabel('Mean Interest Rate')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


##Term analysis 

def plot_term_analysis():
    mean_term_rate = inner_join_df.groupby("term")["int_rate"]

    plt.figure(figsize=(10, 5))
    sns.boxplot(x=inner_join_df["term"], y=inner_join_df["int_rate"])
    plt.title('Interest Rate Distribution by Loan Term')
    plt.xlabel('Loan Term (Months)')
    plt.ylabel('Interest Rate')
    plt.grid(True)
    plt.show()

######################################################################################

def plot_interest_rate_over_time():
    ##apply int_rate["issue_year"] CHECK aline with FED or not
    # Convert 'issue_date' and 'date' columns to datetime format
    # Convert columns to datetime
    inner_join_df["issue_date"] = pd.to_datetime(inner_join_df["issue_date"], errors='coerce')
    df_fed["DATE"] = pd.to_datetime(df_fed["DATE"], errors='coerce')

    # Filter FEDFUNDS data
    df_fed_match = df_fed[(df_fed["DATE"] >= '2012-08-01') & (df_fed["DATE"] <= '2019-12-01')]

    # Check for NaT values
    print(inner_join_df['issue_date'].isna().sum())
    print(df_fed_match['DATE'].isna().sum())

    mean_time_rate = inner_join_df.groupby("issue_date")["int_rate"].mean().reset_index()

    # No normalization of interest rates
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot the mean loan interest rates
    plt.plot(mean_time_rate["issue_date"], mean_time_rate["int_rate"] *100, 
            label='Mean Loan Interest Rate', color='blue')

    # Plot the FED interest rates without normalization
    plt.plot(df_fed_match["DATE"], df_fed_match["FEDFUNDS"], 
            label='FED Interest Rate', color='orange')

    # Add titles and labels
    plt.title('Interest Rates Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Interest Rate (%)', fontsize=12)

    # Add a legend
    plt.legend()

    # Enable grid
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

######################################################################################

def plot_normalised_interest_rate_over_time():

    inner_join_df["issue_date"] = pd.to_datetime(inner_join_df["issue_date"], errors='coerce')
    df_fed["DATE"] = pd.to_datetime(df_fed["DATE"], errors='coerce')
    df_fed_match = df_fed[(df_fed["DATE"] >= '2012-08-01') & (df_fed["DATE"] <= '2019-12-01')]

    mean_time_rate = inner_join_df.groupby("issue_date")["int_rate"].mean().reset_index()

    # Normalize the int_rate using Min-Max normalization
    min_int_rate = mean_time_rate["int_rate"].min()
    max_int_rate = mean_time_rate["int_rate"].max()
    mean_time_rate['normalized_int_rate'] = (mean_time_rate['int_rate'] - min_int_rate) / (max_int_rate - min_int_rate)

    # Normalize the FEDFUNDS
    min_fed = df_fed_match['FEDFUNDS'].min()
    max_fed = df_fed_match['FEDFUNDS'].max()
    df_fed_match['normalized_fed'] = (df_fed_match['FEDFUNDS'] - min_fed) / (max_fed - min_fed)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot the normalized loan interest rates
    plt.plot(mean_time_rate["issue_date"], mean_time_rate["normalized_int_rate"], 
            label='Normalized Loan Interest Rate', color='blue')

    # Plot the normalized FED interest rates
    plt.plot(df_fed_match["DATE"], df_fed_match["normalized_fed"], 
            label='Normalized FED Interest Rate', color='orange')

    # Add titles and labels
    plt.title('Normalized Interest Rates Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Rate', fontsize=12)

    # Add a legend
    plt.legend()

    # Enable grid
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



######################################################################################

# Step 1: Convert grades (A-G) to numerical scores (A=1, G=7)
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
inner_join_df['grade_score'] = inner_join_df['grade'].map(grade_mapping)

# Step 2: Prepare features (X) and target (y)
X = inner_join_df[['annual_inc', 'avg_cur_bal', 'grade_score']]  # Features
y = inner_join_df['int_rate']  # Target variable

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Optionally, display coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

######################################################################################

# Setting up the main window
root = tk.Tk()
root.title("Loan Data Visualization")
root.geometry("600x400") 

def calculate_interest_rate():
    try:
        annual_inc = float(entry_annual_inc.get())
        avg_cur_bal = float(entry_avg_cur_bal.get())
        grade_score = float(entry_grade_score.get())
        
        # Calculate interest rate using coefficients
        int_rate = (-7.601290e-10 * annual_inc +
                    -1.044227e-08 * avg_cur_bal +
                    0.03740701 * grade_score)
        
        # Display the result
        messagebox.showinfo("Interest Rate", f"Calculated Interest Rate: {int_rate:.4f}")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")


# Frame for input fields
input_frame = ttk.LabelFrame(root, text="Interest Rate Calculation", padding=(10, 10))
input_frame.pack(padx=10, pady=10, fill="both", expand="yes")

# Annual Income input
ttk.Label(input_frame, text="Annual Income:").grid(column=0, row=0, padx=5, pady=5, sticky="w")
entry_annual_inc = ttk.Entry(input_frame)
entry_annual_inc.grid(column=1, row=0, padx=5, pady=5)

# Average Current Balance input
ttk.Label(input_frame, text="Average Current Balance:").grid(column=0, row=1, padx=5, pady=5, sticky="w")
entry_avg_cur_bal = ttk.Entry(input_frame)
entry_avg_cur_bal.grid(column=1, row=1, padx=5, pady=5)

# Grade Score input
ttk.Label(input_frame, text="Grade Score:").grid(column=0, row=2, padx=5, pady=5, sticky="w")
entry_grade_score = ttk.Entry(input_frame)
entry_grade_score.grid(column=1, row=2, padx=5, pady=5)

# Button to calculate interest rate
calculate_button = tk.Button(input_frame, text="Calculate Interest Rate", command=calculate_interest_rate)
calculate_button.grid(columnspan=2, pady=10)

# Frame for Customer Analysis
customer_frame = ttk.LabelFrame(root, text="Customer Analysis")
customer_frame.pack(padx=10, pady=10, fill="both", expand="yes")

# Dropdown menu for selecting customer analysis plot type
customer_plot_selection = ttk.Combobox(customer_frame, values=[
    "Loan Amount to Annual Income Ratio",
    "Annual Installment to Annual Income",
    "Total Interest Analysis",
    "Loan Purpose Count",
    "Count by State"
])
customer_plot_selection.set("Select Customer Plot Type")
customer_plot_selection.pack(pady=10)

# Frame for Interest Rate Analysis
interest_rate_frame = ttk.LabelFrame(root, text="Interest Rate Analysis")
interest_rate_frame.pack(padx=10, pady=10, fill="both", expand="yes")

# Dropdown menu for selecting interest rate analysis plot type
interest_rate_plot_selection = ttk.Combobox(interest_rate_frame, values=[
    "Mean Interest Rate by Grade",
    "Mean Interest Rate by Home Ownership",
    "Interest Rate Distribution by Loan Term",
    "Interest Rates Over Time",
    "Normalized Interest Rates Over Time"
])
interest_rate_plot_selection.set("Select Interest Rate Plot Type")
interest_rate_plot_selection.pack(pady=10)

# Button to generate plot
def generate_plot():
    customer_plot_type = customer_plot_selection.get()
    interest_rate_plot_type = interest_rate_plot_selection.get()

    if customer_plot_type != "Select Customer Plot Type":
        if customer_plot_type == "Loan Amount to Annual Income Ratio":
            plot_loan_income_ratio()
        elif customer_plot_type == "Annual Installment to Annual Income":
            plot_installment_analysis()
        elif customer_plot_type == "Total Interest Analysis":
            plot_total_interest()
        elif customer_plot_type == "Loan Purpose Count":
            plot_loan_purpose()
        elif customer_plot_type == "Count by State":
            plot_state_counts()
    
    if interest_rate_plot_type != "Select Interest Rate Plot Type":
        if interest_rate_plot_type == "Mean Interest Rate by Grade":
            plot_grade_analysis()
        elif interest_rate_plot_type == "Mean Interest Rate by Home Ownership":
            plot_home_owership_analysis()
        elif interest_rate_plot_type == "Interest Rate Distribution by Loan Term":
            plot_term_analysis()
        elif interest_rate_plot_type == "Interest Rates Over Time":
            plot_interest_rate_over_time()
        elif interest_rate_plot_type == "Normalized Interest Rates Over Time":
            plot_normalised_interest_rate_over_time()
    
    if customer_plot_type == "Select Customer Plot Type" and interest_rate_plot_type == "Select Interest Rate Plot Type":
        print("Please select a valid plot type.")

plot_button = tk.Button(root, text="Generate Plot", command=generate_plot)
plot_button.pack(pady=10)

# Start the GUI event loop
root.mainloop()
