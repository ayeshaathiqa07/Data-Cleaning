# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset directly
df = sns.load_dataset('titanic')

# Display first 5 rows
display(df.head())

# Show column details
df.info()

# %%
%pip install seaborn pandas numpy matplotlib ipykernel

# %%
# 1. Fill missing 'age' with median age
df['age'] = df['age'].fillna(df['age'].median())

# 2. Fill missing 'embarked' with the mode (most common port)
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 3. Drop 'deck' column (too many missing values)
if 'deck' in df.columns:
    df = df.drop(columns=['deck'])

# 4. Remove duplicate rows
df = df.drop_duplicates()

print("--- Missing Values After Cleaning ---")
print(df.isnull().sum())

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Survival rate by Gender
sns.barplot(x='sex', y='survived', data=df, ax=axes[0], palette='Blues_d')
axes[0].set_title('Survival Rate by Gender')

# Plot 2: Age Distribution
sns.histplot(df['age'], kde=True, ax=axes[1], color='teal')
axes[1].set_title('Passenger Age Distribution')

# Plot 3: Survival count by Passenger Class
sns.countplot(x='pclass', hue='survived', data=df, ax=axes[2], palette='Set2')
axes[2].set_title('Survival Count by Class')

plt.tight_layout()
plt.show()

# %%
# Export cleaned data to CSV
df.to_csv('cleaned_titanic_data.csv', index=False)
print("Cleaned dataset successfully saved as 'cleaned_titanic_data.csv'!")


