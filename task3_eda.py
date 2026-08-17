import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Load the cleaned dataset
df = pd.read_csv('cleaned_titanic_data.csv')

# 2. Statistical Summaries
print("=== DATASET OVERVIEW & STATISTICAL SUMMARY ===")
print("\n--- Numerical Summary ---")
print(df.describe())

print("\n--- Categorical Summary ---")
print(df.describe(include=['O', 'category']))

print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# 3. Correlation Matrix (Numerical Columns)
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Titanic Features')
plt.tight_layout()
plt.show()

# 4. Multi-Plot EDA Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Survival Rate by Sex
sns.barplot(x='sex', y='survived', data=df, ax=axes[0, 0], palette='Blues_d')
axes[0, 0].set_title('Survival Rate by Gender')
axes[0, 0].set_ylabel('Survival Rate')

# Chart 2: Survival Rate by Passenger Class
sns.barplot(x='pclass', y='survived', data=df, ax=axes[0, 1], palette='Greens_d')
axes[0, 1].set_title('Survival Rate by Passenger Class')
axes[0, 1].set_ylabel('Survival Rate')

# Chart 3: Age Distribution by Survival Status
sns.kdeplot(data=df, x='age', hue='survived', common_norm=False, fill=True, ax=axes[1, 0], palette='Set1')
axes[1, 0].set_title('Age Distribution by Survival Status')

# Chart 4: Fare Distribution across Passenger Classes
sns.boxplot(x='pclass', y='fare', data=df, ax=axes[1, 1], palette='Purples')
axes[1, 1].set_title('Fare Distribution across Passenger Classes')
axes[1, 1].set_ylim(0, 150) # Limit y-axis for better visibility

plt.tight_layout()
plt.show()

print("\n=== KEY EDA INSIGHTS ===")
print("1. Females had a significantly higher survival rate than males.")
print("2. 1st Class passengers had the highest survival probability.")
print("3. Passengers who paid higher fares generally belonged to 1st class and had better survival outcomes.")