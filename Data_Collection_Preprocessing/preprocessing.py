import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from ucimlrepo import fetch_ucirepo

print("===================================")
print(" DATA COLLECTION & PREPROCESSING")
print("===================================")

try:

    # Create dataset folder automatically
    os.makedirs("dataset", exist_ok=True)

    # Collect dataset
    print("\nDownloading Adult Census Dataset...")

    adult = fetch_ucirepo(id=2)

    # Combine features and target
    df = pd.concat(
        [adult.data.features, adult.data.targets],
        axis=1
    )

    print("\nDataset collected successfully!")

    # Display shape
    print("\nDataset Shape:")
    print(df.shape)

    # Display first five records
    print("\nFirst Five Records:")
    print(df.head())

    # Save dataset
    df.to_csv(
        "dataset/raw_dataset.csv",
        index=False
    )

    print("\nRaw dataset saved successfully!")

    print("\nFile Location:")
    print(os.path.abspath("dataset/raw_dataset.csv"))

except Exception as e:

    print("\nERROR OCCURRED!")
    print(e)

    # STEP 2: DATA EXPLORATION

print("\n===================================")
print("       DATASET EXPLORATION")
print("===================================")

# 1. Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# 2. Column names
print("\nColumn Names:")
print(df.columns.tolist())

# 3. First five records
print("\nFirst Five Records:")
print(df.head())

# 4. Dataset information
print("\nDataset Information:")
df.info()

# 5. Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# 6. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 7. Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# STEP 3: HANDLE MISSING VALUES

print("\n===================================")
print("       MISSING VALUE HANDLING")
print("===================================")

# Replace '?' with NaN if present
df.replace("?", np.nan, inplace=True)

# Display missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill categorical missing values using mode
categorical_columns = df.select_dtypes(
    include=['object', 'str']
).columns

for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Fill numerical missing values using median
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Display missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nMissing Values Handled Successfully!")

# STEP 4: REMOVE DUPLICATES

print("\n===================================")
print("       DUPLICATE REMOVAL")
print("===================================")

# Count duplicates before removal
duplicates_before = df.duplicated().sum()

print("\nDuplicate Records Before Cleaning:")
print(duplicates_before)

# Remove duplicate records
df = df.drop_duplicates()

# Count duplicates after removal
duplicates_after = df.duplicated().sum()

print("\nDuplicate Records After Cleaning:")
print(duplicates_after)

print("\nRecords Remaining:")
print(len(df))

print("\nDuplicate Records Removed Successfully!")

# STEP 5: DATA TYPE CORRECTION

print("\n===================================")
print("       DATA TYPE CORRECTION")
print("===================================")

# Numerical columns
numeric_columns = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]

# Convert numerical columns
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Categorical columns
categorical_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

# Convert categorical columns
for col in categorical_columns:
    df[col] = df[col].astype('category')

print("\nCorrected Data Types:")

print(df.dtypes)

print("\nData Type Correction Completed Successfully!")
# STEP 6: OUTLIER DETECTION AND HANDLING

print("\n===================================")
print("    OUTLIER DETECTION & HANDLING")
print("===================================")

# Select numerical columns
outlier_columns = [
    'age',
    'fnlwgt',
    'hours-per-week'
]

for col in outlier_columns:

    # Calculate Q1 and Q3
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Count outliers before handling
    outliers = (
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ).sum()

    print(f"\nColumn: {col}")
    print("Outliers Before Handling:", outliers)

    # Cap outliers
    df[col] = df[col].clip(
        lower_limit,
        upper_limit
    )

    print("Outliers Handled Successfully!")

print("\nOutlier Handling Completed!")

# STEP 7: FEATURE ENGINEERING

print("\n===================================")
print("       FEATURE ENGINEERING")
print("===================================")

# Create age groups
df['age_group'] = pd.cut(
    df['age'],
    bins=[0, 25, 40, 60, 100],
    labels=['Young', 'Adult', 'Middle Age', 'Senior']
)

# Categorize working hours
df['work_category'] = pd.cut(
    df['hours-per-week'],
    bins=[0, 20, 40, 100],
    labels=['Part-time', 'Standard', 'Overtime']
)

# Check whether a person has capital gain
df['has_capital_gain'] = np.where(
    df['capital-gain'] > 0, 1, 0
)

print("\nNew Features Created:")
print(df[['age_group', 'work_category', 'has_capital_gain']].head())

print("\nFeature Engineering Completed Successfully!")# STEP 8: SAVE CLEANED DATASET

print("\n===================================")
print("       SAVE CLEANED DATASET")
print("===================================")

# Create dataset folder if it does not exist
os.makedirs("dataset", exist_ok=True)

# Save cleaned dataset
df.to_csv("dataset/cleaned_dataset.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nCleaned Dataset Location:")
print("dataset/cleaned_dataset.csv")
# STEP 8: SAVE CLEANED DATASET

print("\n===================================")
print("       SAVE CLEANED DATASET")
print("===================================")

# Create dataset folder if it does not exist
os.makedirs("dataset", exist_ok=True)

# Save cleaned dataset
# Remove duplicates created during outlier handling
df = df.drop_duplicates().copy()

# Save cleaned dataset
df.to_csv("dataset/cleaned_dataset.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nCleaned Dataset Location:")
print("dataset/cleaned_dataset.csv")

# STEP 9: DATA VISUALIZATION

print("\n===================================")
print("       DATA VISUALIZATION")
print("===================================")

os.makedirs("screenshots", exist_ok=True)

# Chart 1: Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("screenshots/age_distribution.png")
plt.show()

print("Age Distribution Chart Saved!")

# Chart 2: Income Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='income')
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.savefig("screenshots/income_distribution.png")
plt.show()

print("Income Distribution Chart Saved!")

# Chart 3: Work Category
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='work_category')
plt.title("Work Category Distribution")
plt.xlabel("Work Category")
plt.ylabel("Count")
plt.savefig("screenshots/work_category.png")
plt.show()

print("Work Category Chart Saved!")

print("\nData Visualization Completed Successfully!")