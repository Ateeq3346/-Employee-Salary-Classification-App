import streamlit as st
import pandas as pd
import pickle
import numpy as np # Needed for np.nan

# --- Configuration ---
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# --- Load Model and Preprocessing Components ---
# Load the dictionary containing the model, imputer, encoder, and column names
try:
    with open("salary_prediction_components.pkl", "rb") as file:
        components = pickle.load(file)
    model = components['model']
    imputer = components['imputer']
    encoder = components['encoder']
    final_train_columns = components['final_train_columns']
    categorical_cols_fitted = components['categorical_cols_fitted'] # Original categorical column names
    st.success("Model and preprocessing components loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'salary_prediction_components.pkl' not found. Please run 'prepare_model.py' first.")
    st.stop() # Stop the app if components aren't found
except KeyError as e:
    st.error(f"Error loading components: Missing key {e} in 'salary_prediction_components.pkl'. Please regenerate the model components.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading components: {e}")
    st.stop()


# --- Preprocessing Function for new data ---
def preprocess_input_data(df_raw, imputer_obj, encoder_obj, trained_columns, original_categorical_features):
    """
    Applies the same preprocessing steps as the training data:
    1. Replaces '?' with NaN.
    2. Handles and removes any 'income' or one-hot encoded 'income' columns.
    3. Imputes missing categorical values using the fitted imputer.
    4. One-hot encodes categorical features using the fitted encoder.
    5. Aligns columns to match the trained model's expected features.
    """
    df = df_raw.copy()

    # 1. Replace '?' with NaN if any are present (important for raw uploads)
    df.replace('?', np.nan, inplace=True)

    # 2. Handle original 'income' column if present in raw input (it's a target, not a feature)
    if 'income' in df.columns:
        df = df.drop(columns=['income'])
    # Also drop any one-hot encoded income columns if they somehow appeared
    if "income_<=50K" in df.columns:
        df = df.drop(columns=["income_<=50K"])
    if "income_>50K" in df.columns:
        df = df.drop(columns=["income_>50K"])

    # Filter categorical columns that are actually present in the current df
    # and were part of the original categorical features used for fitting
    cols_to_process = [col for col in original_categorical_features if col in df.columns]

    # 3. Impute missing categorical values
    if cols_to_process:
        df[cols_to_process] = imputer_obj.transform(df[cols_to_process])

    # 4. One-hot encode categorical features
    if cols_to_process:
        encoded_features = encoder_obj.transform(df[cols_to_process])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder_obj.get_feature_names_out(cols_to_process), index=df.index)
        # Drop original categorical columns and concatenate encoded ones
        df = pd.concat([df.drop(columns=cols_to_process), encoded_df], axis=1)

    # 5. Align columns to match the trained model's expected features
    # This will add missing columns (from trained_columns) and fill with 0,
    # and drop extra columns (not in trained_columns).
    df_processed = df.reindex(columns=trained_columns, fill_value=0)

    return df_processed

# --- Sidebar Inputs for Single Prediction ---
st.sidebar.header("Input Employee Details")

# Define all original feature columns and their types/options
# These options must match the categories present in your training data
# Numerical Inputs
age = st.sidebar.slider("Age", 17, 90, 30)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 1500000, 200000)
educational_num = st.sidebar.slider("Education Years (educational-num)", 1, 16, 10)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

# Categorical Inputs (ensure options match original dataset values)
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
workclass = st.sidebar.selectbox("Workclass", workclass_options)

education_options = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']
education = st.sidebar.selectbox("Education", education_options)

marital_status_options = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)

occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
occupation = st.sidebar.selectbox("Occupation", occupation_options)

relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
relationship = st.sidebar.selectbox("Relationship", relationship_options)

race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
race = st.sidebar.selectbox("Race", race_options)

gender_options = ['Male', 'Female']
gender = st.sidebar.selectbox("Gender", gender_options)

native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Holand-Netherlands']
native_country = st.sidebar.selectbox("Native Country", native_country_options)


# Build input DataFrame for single prediction (raw format, matching original dataset)
input_data_raw = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
}
input_df_raw = pd.DataFrame(input_data_raw)

st.write("### 🔎 Input Data")
st.write(input_df_raw)

# Predict button for single input
if st.button("Predict Salary Class"):
    # Preprocess the single input DataFrame using the loaded components
    input_df_processed = preprocess_input_data(input_df_raw, imputer, encoder, final_train_columns, categorical_cols_fitted)

    prediction = model.predict(input_df_processed)
    prediction_label = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"✅ Predicted Salary Class: **{prediction_label}**")

# --- Batch Prediction ---
st.markdown("---")
st.markdown("#### 📂 Batch Prediction (Upload CSV)")
st.info("Please upload a CSV file that contains the following original columns (without an 'income' column): `age`, `workclass`, `fnlwgt`, `education`, `educational-num`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data_raw = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data_raw.head())

    # Preprocess the batch data
    try:
        batch_data_processed = preprocess_input_data(batch_data_raw.copy(), imputer, encoder, final_train_columns, categorical_cols_fitted)

        # Make predictions
        batch_preds = model.predict(batch_data_processed)

        # Add predictions to the original batch data (for easier interpretation)
        batch_data_raw['Predicted_Income_>50K'] = batch_preds
        batch_data_raw['Predicted_Income_Class'] = batch_data_raw['Predicted_Income_>50K'].apply(lambda x: '>50K' if x == 1 else '<=50K')

        st.write("✅ Predictions (first 5 rows):")
        st.write(batch_data_raw.head())

        # Download button
        csv = batch_data_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions CSV",
            csv,
            file_name='predicted_salary_classes.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Error during batch prediction: {e}")
        st.warning("Please ensure your uploaded CSV matches the expected column structure and data types.")

