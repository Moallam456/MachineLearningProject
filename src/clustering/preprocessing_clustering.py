import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

print("--- STARTING DATA CLEANING PIPELINE ---")


df = pd.read_csv('H2.csv') 
df = df.drop_duplicates()

# Drop leakage variables 
columns_to_drop = ['ReservationStatus', 'ReservationStatusDate', 'IsCanceled', 'Country', 'Agent', 'Company']
df = df.drop(columns=columns_to_drop, errors='ignore')

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
# Strip hidden whitespace from strings and map to integers 1-12
df['ArrivalDateMonth'] = df['ArrivalDateMonth'].astype(str).str.strip().map(month_map)

# Identify numerical and categorical columns automatically
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Strip hidden whitespace from all remaining categorical columns
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()


# Numerical Pipeline: Fills missing numbers with median, then normalizes scale for K-Means
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Fills missing text, then applies Label Encoding (0, 1, 2...)
# Note: OrdinalEncoder is the Pipeline version of LabelEncoder.
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder()) 
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

print("Processing matrix: imputing missing values, scaling, and encoding...")
X_clean_matrix = preprocessor.fit_transform(df)

# Reconstruct the column headers in the exact order they were processed
final_columns = num_cols + cat_cols

# Convert the raw NumPy math array back into a pandas DataFrame
df_clean = pd.DataFrame(X_clean_matrix, columns=final_columns)

# Export the clean, 25-column pure numerical matrix
df_clean.to_csv('H2_cleaned_for_clustering.csv', index=False)

print(f"Success! Clean matrix generated with {df_clean.shape[1]} pure numerical columns.")
print("Saved as 'H2_cleaned_for_clustering.csv'. You are now ready to run K-Means.")
df.info()
df.head()