from deepchecks.tabular import Dataset
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from deepchecks.tabular.suites import data_integrity
from joblib import dump
import pandas as pd

# Load sklearn dataframe
def load_diabetes_df():
    diabetes = load_diabetes(as_frame=True)
    diabetes_df = diabetes.frame
    return diabetes_df

# Split Dataframe
def split_dataframe():
    diabetes_df = load_diabetes_df()

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(diabetes_df, test_size=0.33, random_state=42)
    return train_df, test_df

# Training Linear Model
def train_linear_model(train, test, filename='trained_diabetes_linear_model'):
    try:
        # Check data types and valid test_frac value
        assert isinstance(train, pd.DataFrame), "train must be a DataFrame"
        assert isinstance(test, pd.DataFrame), "test must be a DataFrame"
        assert isinstance(filename, str), "Filename must be a string"

        # Instantiate a Linear Regression model
        model = LinearRegression()

        # Fit the model with training data
        model.fit(train, train['target'])  # Assuming 'target' is the column to predict

        # Save the trained model to a file
        fname = filename + '.joblib'
        dump(model, fname)

        # Compute R-squared scores for training and test data
        r2_train = model.score(train, train['target'])
        r2_test = model.score(test, test['target'])  # Assuming 'target' is the column to predict

        print("Train R-squared:", r2_train)
        print("Test R-squared:", r2_test)

        # Return scores in a dictionary
        return {'Train-score': r2_train, 'Test-score': r2_test}

    except AssertionError as msg:
        print(msg)
        return msg

# Deepcheck Data integrity check and saving the report into HTML
def data_integrity_check():
    diabetes_df = load_diabetes_df()

    # Run Suite:
    integ_suite = data_integrity()
    suite_result = integ_suite.run(diabetes_df)

    # Save the result report as an HTML file
    suite_result.save_as_html("data_integrity_report.html")
