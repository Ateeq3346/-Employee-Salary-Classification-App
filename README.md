# Employee Salary Classification App

This project contains a Streamlit web application that predicts whether an employee's salary is greater than $50K or less than or equal to $50K based on their demographic and employment data.

The application uses a pre-trained Random Forest Classifier model.

## Project Structure

-   `app.py`: The main Streamlit application file.
-   `prepare_model.py`: A script to preprocess the raw data and train the classification model.
-   `adult 3.csv`: The raw dataset used for training the model.
-   `requirements.txt`: A file listing the Python dependencies for this project.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Prepare the model:**
    First, you need to run the `prepare_model.py` script to train the model and generate the `salary_prediction_components.pkl` file. Make sure you have the `adult 3.csv` dataset in the same directory.
    ```bash
    python prepare_model.py
    ```

2.  **Run the Streamlit application:**
    Once the model components are created, you can start the Streamlit app.
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

## Usage

-   **Single Prediction:** Use the sidebar controls to input an individual's details and click "Predict Salary Class".
-   **Batch Prediction:** Upload a CSV file with the same structure as the original dataset (without the 'income' column) to get predictions for multiple employees at once.
