import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Added this line

print("--- Starting Data Preparation and Model Training ---")

# --- 1. Load the raw dataset ---
# Assuming 'adult 3.csv' is in the same directory as this script.
try:
    df_raw = pd.read_csv('adult 3.csv')
    print("Successfully loaded 'adult 3.csv'")
except FileNotFoundError:
    print("Error: 'adult 3.csv' not found. Please make sure it's in the same directory.")
    exit()

# --- 2. Initial Data Cleaning: Replace '?' with NaN ---
df_raw.replace('?', np.nan, inplace=True)
print("Replaced '?' with NaN in the dataset.")

# --- 3. Split data into training and test sets (before extensive preprocessing) ---
# This ensures a clean split of the original data.
df_train_raw, df_test_raw = train_test_split(df_raw, test_size=0.25, random_state=42)
print(f"Data split into training ({len(df_train_raw)} rows) and test ({len(df_test_raw)} rows) sets.")

# Save these cleaned raw splits for reference and for the Streamlit app to fit preprocessors
df_train_raw.to_csv('adult_train_cleaned.csv', index=False)
df_test_raw.to_csv('adult_test_cleaned.csv', index=False)
print("Saved 'adult_train_cleaned.csv' and 'adult_test_cleaned.csv'")

# --- 4. Identify Feature and Target Columns ---
# 'income' is our target variable. All others are features.
target_column = 'income'
numerical_cols = df_train_raw.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_train_raw.select_dtypes(include='object').columns.tolist()

# Remove the target column from feature lists
if target_column in numerical_cols:
    numerical_cols.remove(target_column)
if target_column in categorical_cols:
    categorical_cols.remove(target_column)

print(f"Numerical features: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")
print(f"Target column: {target_column}")

# --- 5. Preprocessing Pipeline ---

# Initialize Imputer for categorical features (mode imputation)
# Fit on the training data only
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(df_train_raw[categorical_cols])

# Initialize OneHotEncoder
# handle_unknown='ignore' allows the encoder to handle categories not seen during fit
# sparse_output=False ensures a dense array output
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df_train_raw[categorical_cols]) # Fit on training data only

# Function to apply preprocessing consistently
def apply_preprocessing(df, imputer_obj, encoder_obj, numerical_features, categorical_features):
    df_processed = df.copy()

    # Impute categorical features
    if categorical_features:
        df_processed[categorical_features] = imputer_obj.transform(df_processed[categorical_features])

    # One-hot encode categorical features
    if categorical_features:
        encoded_features = encoder_obj.transform(df_processed[categorical_features])
        # Create a DataFrame from encoded features with correct column names
        encoded_df = pd.DataFrame(encoded_features, columns=encoder_obj.get_feature_names_out(categorical_features), index=df_processed.index)
        # Drop original categorical columns and concatenate encoded ones
        df_processed = pd.concat([df_processed.drop(columns=categorical_features), encoded_df], axis=1)

    return df_processed

# Apply preprocessing to training and test sets
X_train_processed = apply_preprocessing(df_train_raw.drop(columns=[target_column]), imputer, encoder, numerical_cols, categorical_cols)
X_test_processed = apply_preprocessing(df_test_raw.drop(columns=[target_column]), imputer, encoder, numerical_cols, categorical_cols)

y_train = df_train_raw[target_column].apply(lambda x: 1 if x.strip() == '>50K' else 0)
y_test = df_test_raw[target_column].apply(lambda x: 1 if x.strip() == '>50K' else 0)

print("Applied imputation and one-hot encoding to training and test features.")

# --- 6. Align Columns (Crucial Step) ---
# Ensure X_test_processed has the exact same columns as X_train_processed
# This handles cases where test set might miss some categories present in train
# or vice-versa (though handle_unknown='ignore' helps with unseen categories in test).
# This also ensures consistent column order.
final_train_columns = X_train_processed.columns.tolist()
X_test_processed = X_test_processed.reindex(columns=final_train_columns, fill_value=0)
print("Aligned columns for training and test sets.")

# Save the final encoded datasets for reference
X_train_processed.to_csv('adult_train_encoded_final.csv', index=False)
X_test_processed.to_csv('adult_test_encoded_final.csv', index=False)
print("Saved 'adult_train_encoded_final.csv' and 'adult_test_encoded_final.csv'")

# --- 7. Train the RandomForestClassifier Model ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train_processed, y_train)
print("RandomForestClassifier model trained successfully.")

# --- 8. Evaluate the Model ---
y_pred = model.predict(X_test_processed)
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- 9. Save all necessary components ---
# Save the trained model, imputer, encoder, and the list of final feature columns
# This ensures that the Streamlit app can load everything it needs for consistent predictions.
components_to_save = {
    'model': model,
    'imputer': imputer,
    'encoder': encoder,
    'final_train_columns': final_train_columns,
    'categorical_cols_fitted': categorical_cols # Store original categorical column names
}

try:
    with open('salary_prediction_components.pkl', 'wb') as file:
        pickle.dump(components_to_save, file)
    print("\nAll model components (model, imputer, encoder, columns) saved to 'salary_prediction_components.pkl'")
except Exception as e:
    print(f"\nError saving components: {e}")

print("\n--- Data Preparation and Model Training Complete ---")
print("You can now run the Streamlit app using 'streamlit run app.py'")
