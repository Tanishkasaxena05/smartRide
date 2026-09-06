import pandas as pd
#load Data
df = pd.read_csv("ncr_ride_bookings.csv")

print(df.head())


# Check dataset size
print("Shape:", df.shape)

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Show column names
print("\nColumns:")
print(df.columns.tolist())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Cleaned column names:")
print(df.columns.tolist())

# Check duplicate rows
duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

# Remove duplicates
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Clean spaces from text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

print("\nText spaces cleaned successfully.")

# Check unique values in categorical columns
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(20))

# Check missing values
missing = df.isnull().sum()

print("\nMissing values:")
print(missing[missing > 0].sort_values(ascending=False))

# Check booking status distribution
print("\nBooking Status:")
print(df["booking_status"].value_counts(dropna=False))

# Check missing values by booking status
print("\nMissing values by booking status:")

for col in df.columns:
    missing_by_status = df.groupby("booking_status")[col].apply(
        lambda x: x.isnull().sum()
    )
    
    if missing_by_status.sum() > 0:
        print(f"\n{col}:")
        print(missing_by_status)

# Convert date/time columns
datetime_columns = [
    "date",
    "time"
]

for col in datetime_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

print("\nDate/time columns converted.")
print(df.dtypes)

# Create useful date/time features

if "date" in df.columns:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name()

if "time" in df.columns:
    df["hour"] = df["time"].dt.hour

print("\nNew date/time features created.")

print(df[[
    col for col in ["date", "time", "year", "month", "day", "day_name", "hour"]
    if col in df.columns
]].head())

# Check numerical columns
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumerical columns:")
print(numeric_columns.tolist())

# Basic statistics
print("\nNumerical data summary:")
print(df[numeric_columns].describe())

# Check negative values in numerical columns

for col in numeric_columns:
    negative_count = (df[col] < 0).sum()
    
    if negative_count > 0:
        print(f"{col}: {negative_count} negative values")

# Check rating columns

rating_columns = [
    "driver_ratings",
    "customer_rating"
]

for col in rating_columns:
    if col in df.columns:
        print(f"\n{col}:")
        print("Minimum:", df[col].min())
        print("Maximum:", df[col].max())
        print("Missing:", df[col].isnull().sum())

# Check invalid rating values

for col in rating_columns:
    if col in df.columns:
        invalid = ((df[col] < 0) | (df[col] > 5)).sum()

        print(f"{col} - Invalid values: {invalid}")

# Standardize categorical columns

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = df[col].str.strip().str.lower()

print("\nCategorical columns standardized.")

# Check important categorical columns
for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

# Check unique categorical values

important_columns = [
    "booking_status",
    "vehicle_type",
    "payment_method",
    "customer_type"
]

for col in important_columns:
    if col in df.columns:
        print(f"\n{col} unique values:")
        print(df[col].dropna().unique())

# Missing value percentage

missing_summary = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (df.isnull().sum() / len(df)) * 100
})

missing_summary = missing_summary[
    missing_summary["missing_count"] > 0
].sort_values("missing_percentage", ascending=False)

print("\nMissing Value Summary:")
print(missing_summary)

# Show missing values with percentage

missing_summary = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2)
})

missing_summary = missing_summary[
    missing_summary["missing_count"] > 0
].sort_values("missing_percentage", ascending=False)

print("\nMissing Value Summary:")
print(missing_summary)

# Check completely empty rows

empty_rows = df.isnull().all(axis=1).sum()

print("\nCompletely empty rows:", empty_rows)

# Remove completely empty rows
df = df.dropna(how="all")

print("Shape after removing empty rows:", df.shape)

# Final duplicate check

final_duplicates = df.duplicated().sum()

print("\nFinal duplicate rows:", final_duplicates)

if final_duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")
else:
    print("No duplicate rows found.")


# Check data types

print("\nData Types:")
print(df.dtypes)

# Count columns by data type
print("\nData type summary:")
print(df.dtypes.value_counts())

# Convert important columns to numeric

numeric_columns_to_convert = [
    "booking_value",
    "ride_distance",
    "driver_ratings",
    "customer_rating",
    "avg_vtat",
    "avg_ctat"
]

for col in numeric_columns_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nNumeric columns converted successfully.")

# Check their data types
for col in numeric_columns_to_convert:
    if col in df.columns:
        print(f"{col}: {df[col].dtype}")

# Check minimum and maximum values

check_columns = [
    "booking_value",
    "ride_distance",
    "driver_ratings",
    "customer_rating",
    "avg_vtat",
    "avg_ctat"
]

print("\nValue Range Check:")

for col in check_columns:
    if col in df.columns:
        print(f"\n{col}")
        print("Minimum:", df[col].min())
        print("Maximum:", df[col].max())

# Check all columns

print("\nAll columns in dataset:")
for i, col in enumerate(df.columns, start=1):
    print(i, ":", col)


# Check duplicate column names

duplicate_columns = df.columns[df.columns.duplicated()].tolist()

print("\nDuplicate column names:", duplicate_columns)

# Check blank strings in text columns

print("\nBlank strings in categorical columns:")

for col in df.select_dtypes(include="object").columns:
    blank_count = df[col].astype("string").str.strip().eq("").sum()
    
    if blank_count > 0:
        print(f"{col}: {blank_count} blank values")

# Convert blank strings to NaN

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

print("\nBlank strings converted to NaN.")

# Final missing value check

final_missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (
        df.isnull().sum() / len(df) * 100
    ).round(2)
})

final_missing = final_missing[
    final_missing["missing_count"] > 0
].sort_values("missing_count", ascending=False)

print("\nFinal Missing Value Summary:")
print(final_missing)

# STEP 31: Save cleaned dataset

import os

# Create cleaned folder if it does not exist
os.makedirs("data/cleaned", exist_ok=True)

# Save cleaned dataset
output_path = "data/cleaned/ncr_ride_bookings_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("File:", output_path)
print("Final shape:", df.shape)

# STEP 32: Final Data Quality Check

print("\n========== FINAL DATA QUALITY CHECK ==========")

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Completely empty rows
print("\nCompletely empty rows:")
print(df.isnull().all(axis=1).sum())

# Total missing values
print("\nTotal missing values:")
print(df.isnull().sum().sum())

# Number of columns
print("\nNumber of columns:")
print(len(df.columns))

# Data types
print("\nData types:")
print(df.dtypes)

print("\n========== CHECK COMPLETE ==========")

print("\n========== CLEANED DATA ==========")

print("\nFirst 10 rows:")
print(df.head(10))

print("\nLast 10 rows:")
print(df.tail(10))

print("\nCleaned dataset shape:")
print(df.shape)



