# Importing necessary libraries
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fetching the dataset
bank_marketing = fetch_ucirepo(id=222)
data = bank_marketing.data.original

# Display the first 5 rows of the dataset
data.head()

# Checking for missing values
data.isna().sum()

# Replacing categorical values with numerical values
data = data.replace({"yes": 1, "no": 0})

# Mapping month names to numerical values
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

# Filling missing values with the mode (most common value)
data['job'] = data['job'].fillna(data['job'].mode()[0])
data['education'] = data['education'].fillna(data['education'].mode()[0])
data['contact'] = data['contact'].fillna('unknown')
data['poutcome'] = data['poutcome'].fillna('unknown')

# Splitting data based on subscription status
subscribed = data[data['y'] == 1]
not_subscribed = data[data['y'] == 0]

# T-test for Age
t_stat_age, p_val_age = stats.ttest_ind(subscribed['age'], not_subscribed['age'])
print("T-test for Age: t-statistic =", t_stat_age, ", p-value =", p_val_age)
if p_val_age < 0.05:
    print("Reject H0 for age")
else:
print("Fail to reject H0 for age")

# Plot distribution for age
plt.figure(figsize=(12, 6), dpi=300)
sns.boxplot(data=data, x='age', y='y')
plt.title('Age Distribution by Subscription Status')
plt.xlabel('Age')
plt.ylabel('Subscription')
plt.show()

# T-test for Duration
t_stat_duration, p_val_duration = stats.ttest_ind(subscribed['duration'], not_subscribed['duration'])
print("T-test for Duration: t-statistic =", t_stat_duration, ", p-value =", p_val_duration)
if p_val_duration < 0.05:
    print("Reject H0 for duration")
else:
    print("Fail to reject H0 for duration")

# Plot distribution for duration
plt.figure(figsize=(12, 6), dpi=300)
sns.boxplot(data=data, x='duration', y='y')
plt.title('Duration Distribution by Subscription Status')
plt.xlabel('Duration')
plt.ylabel('Subscription')
plt.show()

# ANOVA for Job
anova_model = smf.ols('y ~ C(job)', data=data).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("\nANOVA for Job:\n", anova_table)
if anova_table['PR(>F)'].iloc[0] < 0.05:
    print("Reject H0 for job")
else:
    print("Fail to reject H0 for job")

# Plot boxplot for job
plt.figure(figsize=(12, 6), dpi=300)
sns.boxplot(data=data, x='y', y='job', )
plt.title('Subscription Success Rate by Job Category')
plt.xlabel('Success Rate')
plt.ylabel('Job Category')
plt.xticks(rotation=45)
plt.show()

# Chi-square test for Job vs Subscription
contingency_table_job = pd.crosstab(data['job'], data['y'])
chi2_job, p_val_job, dof_job, ex_job = stats.chi2_contingency(contingency_table_job)
print("\nChi-square Test for Job vs y: chi2 =", chi2_job, ", p-value =", p_val_job)
if p_val_job < 0.05:
    print("Reject H0 for job vs y")
else:
    print("Fail to reject H0 for job vs y")

# Chi-square test for Marital Status vs Subscription
contingency_table_marital = pd.crosstab(data['marital'], data['y'])
chi2_marital, p_val_marital, dof_marital, ex_marital = stats.chi2_contingency(contingency_table_marital)
print("\nChi-square Test for Marital vs y: chi2 =", chi2_marital, ", p-value =", p_val_marital)
if p_val_marital < 0.05:
    print("Reject H0 for marital vs y")
else:
    print("Fail to reject H0 for marital vs y")

# Logistic Regression Model
X = data.drop(['y'], axis=1).select_dtypes(include=[np.number])
y = data['y']
X = sm.add_constant(X)  # Add intercept to the model

model = sm.Logit(y, X)
result = model.fit()

# Display model summary
summary = result.summary()

# Extracting and displaying model coefficients and statistics
coefficients = result.params
standard_errors = result.bse
p_values = result.pvalues
odds_ratios = np.exp(coefficients)

results_df = pd.DataFrame({
    'Feature': coefficients.index,
    'Coefficient': coefficients.values,
    'Standard Error': standard_errors.values,
    'p-value': p_values.values,
    'Odds Ratio': odds_ratios.values
})
