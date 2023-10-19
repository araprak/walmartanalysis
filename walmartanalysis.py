import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Importing the csv file and loading it into the datatype df
df = pd.read_csv('Walmart Data Analysis and Forcasting.csv')
    
# What is the total weekly sales for all stores?
total_weekly_sales = df['Weekly_Sales'].sum()
print("Total weekly sales for all stores:", total_weekly_sales)

# What is the average weekly sales for each store?
avg_weekly_sales = df.groupby('Store')['Weekly_Sales'].mean()
print("Average weekly sales for each store:")
print(avg_weekly_sales)

# Which store has the highest weekly sales?
max_weekly_sales = df.groupby('Store')['Weekly_Sales'].sum().idxmax()
print("Store with the highest weekly sales:", max_weekly_sales)

# Which store has the lowest weekly sales?
min_weekly_sales = df.groupby('Store')['Weekly_Sales'].sum().idxmin()
print("Store with the lowest weekly sales:", min_weekly_sales)

# What is the overall average weekly sales for all stores?
overall_avg_sales = df['Weekly_Sales'].mean()
print("Overall average weekly sales for all stores:", overall_avg_sales)

# What is the median weekly sales for all stores?
median_sales = df['Weekly_Sales'].median()
print("Median weekly sales for all stores:", median_sales)

# What is the standard deviation of weekly sales for all stores?
std_dev_sales = df['Weekly_Sales'].std()
print("Standard deviation of weekly sales for all stores:", std_dev_sales)

# What is the minimum weekly sales for all stores?
min_sales = df['Weekly_Sales'].min()
print("Minimum weekly sales for all stores:", min_sales)

# What is the maximum weekly sales for all stores?
max_sales = df['Weekly_Sales'].max()
print("Maximum weekly sales for all stores:", max_sales)

# How many holiday weeks are there in the dataset?
num_holiday_weeks = df[df['Holiday_Flag'] == 1]['Holiday_Flag'].count()
print("Number of holiday weeks:", num_holiday_weeks)

# What is the total weekly sales during holiday weeks?
holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales'].sum()
print("Total weekly sales during holiday weeks:", holiday_sales)

# What is the average weekly sales during holiday weeks?
avg_holiday_sales = df[df['Holiday_Flag'] == 1]['Weekly_Sales'].mean()
print("Average weekly sales during holiday weeks:", avg_holiday_sales)

# What is the total weekly sales during non-holiday weeks?
non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales'].sum()
print("Total weekly sales during non-holiday weeks:", non_holiday_sales)

# What is the average weekly sales during non-holiday weeks?
avg_non_holiday_sales = df[df['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
print("Average weekly sales during non-holiday weeks:", avg_non_holiday_sales)

# What is the correlation between weekly sales and temperature?
corr_sales_temp = df['Weekly_Sales'].corr(df['Temperature'])
print("Correlation between weekly sales and temperature:", corr_sales_temp)

# What is the correlation between weekly sales and fuel price?
corr_sales_fuel = df['Weekly_Sales'].corr(df['Fuel_Price'])
print("Correlation between weekly sales and fuel price:", corr_sales_fuel)

# correlation between weekly sales and CPI

corr_coeff = df['Weekly_Sales'].corr(df['CPI'])
print("Correlation between Weekly_Sales and CPI:", corr_coeff)

# Set figure size
plt.figure(figsize=(8, 6))

# Create histogram
sns.histplot(df['Weekly_Sales'], bins=50, kde=True)

# Set title and labels
plt.title('Distribution of Weekly Sales')
plt.xlabel('Weekly Sales')
plt.ylabel('Count')

# Show plot
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Store', y='Weekly_Sales', data=df)
plt.xlabel('Store', fontsize=7)
plt.ylabel('Weekly Sales', fontsize=14)
plt.title('Weekly Sales by Store', fontsize=16)
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
plt.show()


# Create a scatter plot of Weekly_Sales against Temperature
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df['Temperature'], df['Weekly_Sales'], alpha=0.5)
ax.set_xlabel('Temperature')
ax.set_ylabel('Weekly Sales')
ax.set_title('Scatter Plot of Weekly Sales vs. Temperature')
plt.show()

# Compute the correlation matrix
corr = df[['Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr()

# Create a heatmap of the correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
plt.show()

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Compute the total sales for each week
weekly_sales = df.groupby(['Date'])['Weekly_Sales'].sum().reset_index()

# Create a line plot of weekly sales over time
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales['Date'], weekly_sales['Weekly_Sales'])
plt.title('Weekly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# Convert the "Date" column to a datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Set the "Date" column as the index
df.set_index("Date", inplace=True)

# Resample the data to a monthly frequency and calculate the mean weekly sales for each month
monthly_sales = df["Weekly_Sales"].resample("M").mean()

# Plot the monthly sales data with better visualization
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales, color='blue')
plt.xlabel("Date")
plt.ylabel("Average Weekly Sales")
plt.title("Monthly Average Weekly Sales")
plt.grid(True)
plt.show()

# Group the data by "Holiday_Flag" and calculate the mean weekly sales for each group
holiday_sales = df[df["Holiday_Flag"] == 1]["Weekly_Sales"]
non_holiday_sales = df[df["Holiday_Flag"] == 0]["Weekly_Sales"]
holiday_mean = holiday_sales.mean()
non_holiday_mean = non_holiday_sales.mean()

# Perform a t-test to compare the mean weekly sales between holiday and non-holiday weeks
t_stat, p_val = ttest_ind(holiday_sales, non_holiday_sales)

# Print the results of the t-test
print("t-statistic: {:.2f}".format(t_stat))
print("p-value: {:.4f}".format(p_val))
if p_val < 0.05:
    print("There is a statistically significant difference in sales between holiday and non-holiday weeks.")
else:
    print("There is not a statistically significant difference in sales between holiday and non-holiday weeks.")
