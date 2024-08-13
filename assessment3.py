# Importing necessary libraries
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# Fetching the dataset from UCI Machine Learning Repository
# Dataset ID: 222 (Bank Marketing Data Set)
bank_marketing = fetch_ucirepo(id=222)
data = bank_marketing.data.original

# Display the first 5 rows of the dataset to understand its structure
data.head()

# Checking for missing values in the dataset
# This step helps in identifying any gaps in the data that might need to be addressed
data.isna().sum()

# Replacing categorical values with numerical equivalents
# Here, 'yes' is replaced by 1 and 'no' by 0 for easier analysis and modeling
data = data.replace({"yes": 1, "no": 0})

# Mapping month names to numerical values for easier processing
# E.g., 'jan' is mapped to 1, 'feb' to 2, and so on
months = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}
data['month'] = data['month'].map(months)

# Filling missing values in categorical columns with the most frequent value (mode)
# This ensures that missing data does not cause errors in subsequent analysis
data['job'] = data['job'].fillna(data['job'].mode()[0])
data['education'] = data['education'].fillna(data['education'].mode()[0])
data['contact'] = data['contact'].fillna('unknown')
data['poutcome'] = data['poutcome'].fillna('unknown')

# Splitting the dataset based on subscription status ('y' column)
# This allows for separate analysis of customers who subscribed vs those who didn't
subscribed = data[data['y'] == 1]
not_subscribed = data[data['y'] == 0]

# Performing a T-test to compare the mean age of subscribed vs non-subscribed customers
t_stat_age, p_val_age = stats.ttest_ind(subscribed['age'], not_subscribed['age'])
print("T-test for Age: t-statistic =", t_stat_age, ", p-value =", p_val_age)
if p_val_age < 0.05:
    print("Reject H0 for age: There is a significant difference in age between the two groups.")
else:
    print("Fail to reject H0 for age: No significant difference in age between the two groups.")

# Plotting the distribution of age by subscription status
plt.figure(figsize=(12, 6), dpi=300)
sns.boxplot(data=data, x='y', y='age')
plt.title('Age Distribution by Subscription Status')
plt.xlabel('Subscription Status (0=No, 1=Yes)')
plt.ylabel('Age')
plt.show()

# Performing a T-test to compare the mean duration of subscribed vs non-subscribed customers
t_stat_duration, p_val_duration = stats.ttest_ind(subscribed['duration'], not_subscribed['duration'])
print("T-test for Duration: t-statistic =", t_stat_duration, ", p-value =", p_val_duration)
if p_val_duration < 0.05:
    print("Reject H0 for duration: There is a significant difference in call duration between the two groups.")
else:
    print("Fail to reject H0 for duration: No significant difference in call duration between the two groups.")

# Plotting the distribution of call duration by subscription status
plt.figure(figsize=(12, 6), dpi=300)
sns.boxplot(data=data, x='y', y='duration')
plt.title('Duration Distribution by Subscription Status')
plt.xlabel('Subscription Status (0=No, 1=Yes)')
plt.ylabel('Call Duration (seconds)')
plt.show()

# Performing ANOVA to check if the job type has a significant effect on subscription status
anova_model = smf.ols('y ~ C(job)', data=data).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("\nANOVA for Job:\n", anova_table)
if anova_table['PR(>F)'].iloc[0] < 0.05:
    print("Reject H0 for job: Job type has a significant effect on subscription status.")
else:
    print("Fail to reject H0 for job: Job type does not significantly affect subscription status.")

# Plotting the subscription success rate by job category
plt.figure(figsize=(12, 6), dpi=300)
sns.boxplot(data=data, x='y', y='job')
plt.title('Subscription Success Rate by Job Category')
plt.xlabel('Subscription Status (0=No, 1=Yes)')
plt.ylabel('Job Category')
plt.xticks(rotation=45)
plt.show()

# Performing a Chi-square test to examine the relationship between job type and subscription status
contingency_table_job = pd.crosstab(data['job'], data['y'])
chi2_job, p_val_job, dof_job, ex_job = stats.chi2_contingency(contingency_table_job)
print("\nChi-square Test for Job vs Subscription Status: chi2 =", chi2_job, ", p-value =", p_val_job)
if p_val_job < 0.05:
    print("Reject H0 for job vs y: There is a significant association between job type and subscription status.")
else:
    print("Fail to reject H0 for job vs y: No significant association between job type and subscription status.")

# Performing a Chi-square test to examine the relationship between marital status and subscription status
contingency_table_marital = pd.crosstab(data['marital'], data['y'])
chi2_marital, p_val_marital, dof_marital, ex_marital = stats.chi2_contingency(contingency_table_marital)
print("\nChi-square Test for Marital Status vs Subscription Status: chi2 =", chi2_marital, ", p-value =", p_val_marital)
if p_val_marital < 0.05:
    print("Reject H0 for marital vs y: There is a significant association between marital status and subscription status.")
else:
    print("Fail to reject H0 for marital vs y: No significant association between marital status and subscription status.")

# Building a Logistic Regression model to predict subscription status based on available features
# Excluding the 'y' column from the features (X), and using it as the target variable (y)
X = data.drop(['y'], axis=1).select_dtypes(include=[np.number])
y = data['y']
X = sm.add_constant(X)  # Adding an intercept term to the model

# Fitting the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Displaying the model summary, which includes coefficients, standard errors, p-values, and other statistics
summary = result.summary()

# Extracting and displaying model coefficients and relevant statistics
coefficients = result.params
standard_errors = result.bse
p_values = result.pvalues
odds_ratios = np.exp(coefficients)

# Creating a DataFrame to neatly display the results
results_df = pd.DataFrame({
    'Feature': coefficients.index,
    'Coefficient': coefficients.values,
    'Standard Error': standard_errors.values,
    'p-value': p_values.values,
    'Odds Ratio': odds_ratios.values
})

print("\nLogistic Regression Results:\n", results_df)
